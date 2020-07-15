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
import java.io.IOException;

/** This TensorFlowLite classifier works with the float MobileNet model. */
public class LanguageModelMobileNet extends LanguageModel {


  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][][] labelProbArray = null;

  /**
   * Initializes a {@code LanguageModelMobileNet}.
   *
   * @param activity
   */
  public LanguageModelMobileNet(Service activity, Device device, int numThreads)
      throws IOException {
    super(activity, device, numThreads);
    labelProbArray = new float[1][MAX_HISTORY][getNumLabels()];
  }

  @Override
  protected String getModelPath() {
    return "converted_model.tflite";
  }

  @Override
  protected String getLabelPath() {
    return "labels.txt";
  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][MAX_HISTORY-1][labelIndex];
  }

  @Override
  protected void runInference() {
    tflite.run(intValues, labelProbArray);
  }

}
