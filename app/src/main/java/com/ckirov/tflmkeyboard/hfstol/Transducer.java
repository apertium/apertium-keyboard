package com.ckirov.tflmkeyboard.hfstol;

import java.util.Collection;

public abstract class Transducer {
    abstract Collection<String> analyze(String str) throws NoTokenizationException;
}
