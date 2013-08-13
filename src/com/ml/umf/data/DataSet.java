package com.ml.umf.data;

import java.util.List;

public class DataSet {
    List<Instance> instances;
    
    public List<Instance> getInstances(){
        return instances;
    }
    
    public void setInstances(List<Instance> instances){
        this.instances = instances;
    }
}
