package com.ml.umf.common;

import com.ml.umf.data.Instance;

public interface SGDLearner {
    public void update(Instance instance) throws Exception;  
}
