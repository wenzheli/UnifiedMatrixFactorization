package com.ml.umf.common;

import com.ml.umf.data.Instance;

public interface Classifier {
    public double predict(Instance instance);
}
