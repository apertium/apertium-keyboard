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

package org.apertium.keyboard.hfstol;

import android.app.Service;
import android.content.res.AssetFileDescriptor;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.io.DataInputStream;
import java.nio.channels.FileChannel;
import java.util.*;

import org.apertium.keyboard.hfstol.WeightedTransducer;

import android.util.Log;

public class HfstLanguageModel {

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;
  private static final int MAX_PRED = 12;

  /** History of symbols accepted. */
  protected static final int MAX_HISTORY = 30;

  private WeightedTransducer t;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
//  protected Interpreter hfstol;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
/*
  public static HfstLanguageModel create(Service activity, Model model, Device device, int numThreads)
      throws IOException {
    return new LanguageModelMobileNet(activity, device, numThreads);
  }
*/
  /** Initializes a {@code LanguageModel}. */
/*  protected HfstLanguageModel(Service activity, Device device, int numThreads) throws IOException {
    hfstolModel = loadModelFile(activity);

  }
*/
  public HfstLanguageModel() {
	FileInputStream transducerfile = null;
        try {
	transducerfile = new FileInputStream(getModelPath()); 
	} catch (java.io.FileNotFoundException e) {
        }
        TransducerHeader h = null;
        try {
        h = new TransducerHeader(transducerfile);
        } catch (IOException e) {
        } catch (FormatException e) {
        }
	DataInputStream charstream = new DataInputStream(transducerfile);
	//System.out.println("Reading alphabet...");
        try {
	TransducerAlphabet a = new TransducerAlphabet(charstream, h.getSymbolCount());
    	//System.out.println("Reading transition and index tables...");
            try {
    	t = new WeightedTransducer(transducerfile, h, a);
            } catch (IOException e) {
            }
        } catch(IOException e) { 
        }
  }
  

  /** Memory-map the model file in Assets. */
/*  private MappedByteBuffer loadModelFile(Service activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
*/

  /** Generates up to the next word or morpheme boundary. */
  public HashMap<String, Float> generate(final String history) {
    HashMap<String, Float> results = new HashMap<>();
    try {
      Collection<String> analyses = t.analyze(history);
      for (String analysis : analyses) {
        results.put(analysis, 0.0f);
      }
    } catch( NoTokenizationException e)
    { }

    return results;
  }



  /** Closes the interpreter and model to release resources. */
/*  public void close() {
    if (hfstol != null) {
      hfstol.close();
      hfstol = null;
    }
    hfstolModel = null;
  }
*/

  /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected String getModelPath() {
     return "error.model.hfstol";
  }


}
