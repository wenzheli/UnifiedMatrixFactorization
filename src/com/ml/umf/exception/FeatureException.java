package com.ml.umf.exception;

public class FeatureException extends Exception{
    public FeatureException(String message){
        super(message);
    }
    
    public FeatureException(String message, Throwable e){
        super(message, e);
    }
    
    public FeatureException(Exception ex){
        super(ex);
    }
}
