/* Copyright 2019 Christo Kirov. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.apertium.keyboard.tflite;

import android.app.Service;
import android.content.res.AssetFileDescriptor;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import android.util.Log;

/** A language model that uses TensorFlow Lite. */
public abstract class LanguageModel {

  /** The model type used for classification. */
  public enum Model {
    FLOAT,
    QUANTIZED,
  }

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  }

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;
  private static final int MAX_PRED = 12;

  /** History of symbols accepted. */
  protected static final int MAX_HISTORY = 30;

  /** Special character indeces. */
  private Set<Integer> morpheme_edges;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_ALPHABET_SIZE = 138;

  /** Preallocated buffers for storing word indexe data in. */
  protected final int[][] intValues = new int[1][MAX_HISTORY];

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;
  private Map<String, Integer> s_to_i;
  private Map<Integer, String> i_to_s;

  /** Optional GPU delegate for accleration. */
  private GpuDelegate gpuDelegate = null;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static LanguageModel create(Service activity, Model model, Device device, int numThreads)
      throws IOException {
    return new LanguageModelMobileNet(activity, device, numThreads);
  }

  /** Inner class to store predictions per vocabulary item. */
  public static class Prediction implements Comparable<Prediction>{
    public int id;
    public float logit;
    public String word;

    public int compareTo(Prediction pred){
      if (logit < pred.logit){
        return 1;
      }
      else if (logit == pred.logit) {
        return 0;
      }
      else return -1;
    }
  }

  /** Initializes a {@code LanguageModel}. */
  protected LanguageModel(Service activity, Device device, int numThreads) throws IOException {
    tfliteModel = loadModelFile(activity);
    switch (device) {
      case NNAPI:
        tfliteOptions.setUseNNAPI(true);
        break;
      case GPU:
        gpuDelegate = new GpuDelegate();
        tfliteOptions.addDelegate(gpuDelegate);
        break;
      case CPU:
        break;
    }
    tfliteOptions.setNumThreads(numThreads);
    tflite = new Interpreter(tfliteModel, tfliteOptions);
    labels = loadLabelList(activity);
    s_to_i = new HashMap<String, Integer>();
    i_to_s = new HashMap<Integer, String>();
    morpheme_edges = new HashSet<Integer>();
    for (int i = 0; i < labels.size(); i++) {
      s_to_i.put(labels.get(i), i);
      i_to_s.put(i, labels.get(i));
      if (labels.get(i).endsWith("@") || labels.get(i).endsWith("_") || labels.get(i).endsWith("<eos>")){
        morpheme_edges.add(i);
      }
    }

  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Service activity) throws IOException {
    List<String> labels = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
    String line;
    while ((line = reader.readLine()) != null) {
      labels.add(line);
    }
    reader.close();
    return labels;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Service activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Push history string to preallocated space for TFLite. */
  private void pushHistory(final String histBuffer){
    for (int i = 0; i < MAX_HISTORY; i++){
      String ch = String.valueOf(histBuffer.charAt(i));
      if( s_to_i.containsKey(ch) ) {
        intValues[0][i] = s_to_i.get(ch);
      }
      else if (ch.equals("\n")) {
        intValues[0][i] = s_to_i.get("<eos>");
      } else {
        intValues[0][i] = s_to_i.get("_");
      }
    }
  }

  /** Generates up to the next word or morpheme boundary. */
  public HashMap<String, Float> generate(final String history) {
    // convert string into 30-long integer sequence and put into intbuffer
    final String baseBuffer = "______________________________";
    String histBuffer = history.replace(' ', '_');
    if (histBuffer.length() <= MAX_HISTORY){
      histBuffer = baseBuffer.substring(0, MAX_HISTORY-history.length()) + histBuffer;
      //for (int i = 0; i < (MAX_HISTORY - history.length()); i++){
      //  histBuffer = "_" + histBuffer;
    }
    else {
      histBuffer = histBuffer.substring(histBuffer.length() - MAX_HISTORY);
    }
    //Log.d("HISTORY", histBuffer);
    pushHistory(histBuffer);

    // run inference
    runInference();

    //read out the probabilities, and sort the result to find the top 3...
    PriorityQueue<Prediction> predictions = new PriorityQueue<>();
    for (int i = 0; i < labels.size(); i++){
      Prediction pred = new Prediction();
      pred.logit = getProbability(i);
      pred.id = i;
      pred.word = i_to_s.get(i);
      predictions.add(pred);
    }

    // greedy unroll the top 3 for a while until we hit an edge symbols
    HashMap<String, Float> results = new HashMap<>();
    PriorityQueue<Prediction> tmpPredictions = new PriorityQueue<>();
    for (int i = 0; i < MAX_RESULTS; i++){
      Prediction pred = predictions.poll();
      String predword = pred.word;
      String tmpHistory = histBuffer.substring(1, MAX_HISTORY) + pred.word;
      while(predword.equals("_") || predword.equals("@") || predword.equals("<eos>")
              || !predword.endsWith("_") && !predword.endsWith("@") && !predword.endsWith("<eos>")
              && !(predword.length() > MAX_PRED) ) {
        // push new history and rerun inference
        pushHistory(tmpHistory);
        runInference();
        // find the top scoring item
        tmpPredictions.clear();
        for (int j = 0; j < labels.size(); j++){
          Prediction tmppred = new Prediction();
          tmppred.logit = getProbability(j);
          tmppred.id = j;
          tmppred.word = i_to_s.get(j);
          tmpPredictions.add(tmppred);
        }
        String predchar = tmpPredictions.poll().word;
        predword = predword + predchar;
        tmpHistory = tmpHistory.substring(1, MAX_HISTORY) + predchar;
      }
      results.put(predword, pred.logit);
    }

    return results;
  }



  /** Closes the interpreter and model to release resources. */
  public void close() {
    if (tflite != null) {
      tflite.close();
      tflite = null;
    }
    if (gpuDelegate != null) {
      gpuDelegate.close();
      gpuDelegate = null;
    }
    tfliteModel = null;
  }


  /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected abstract String getModelPath();

  /**
   * Get the name of the label file stored in Assets.
   *
   * @return
   */
  protected abstract String getLabelPath();

  /**
   * Read the probability value for the specified label This is either the original value as it was
   * read from the net's output or the updated value after the filter was applied.
   *
   * @param labelIndex
   * @return
   */
  protected abstract float getProbability(int labelIndex);

  /**
   * Run inference using the prepared input in intValues. Afterwards, the result will be
   * provided by getProbability().
   *
   * <p>This additional method is necessary, because we don't have a common base for different
   * primitive data types.
   */
  protected abstract void runInference();

  /**
   * Get the total number of labels.
   *
   * @return
   */
  protected int getNumLabels() {
    return labels.size();
  }
}
