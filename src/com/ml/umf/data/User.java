package com.ml.umf.data;

import java.util.Random;


public class User extends Entity {
    private double[] latentFeatures;
    private double[] userFeatures;
    private double[] userWeights;
    
    public static final int numOfUserFeatures = 100;
    public static final int numOfItemFeatures = 100;
    
    public static final int numOfLatentFeatures = 20;
   
    public User(){
        super();
        latentFeatures = new double[numOfLatentFeatures];
        // randomly initialize the latent value
        for (int i = 0; i < numOfLatentFeatures; i++){
            latentFeatures[i] = new Random().nextDouble();
        }
        
        userFeatures = new double[numOfUserFeatures];
        userWeights = new double[numOfUserFeatures + numOfItemFeatures];
        
        // randomly initialize the weights
        for (int i = 0; i < userWeights.length; i++){
            userWeights[i] = new Random().nextDouble();
        }
    }
    
    public double[] getLatentFeatures(){
        return latentFeatures;
    }
    
    public void setLatentFeatures(double[] latentFeatures){
        this.latentFeatures = latentFeatures;
    }
    
    public double[] getUserFeatures(){
        return userFeatures;
    }
    
    public void setUserFeatures(double[] userFeatures){
        this.userFeatures = userFeatures;
    }
    
    public void setUserWeights(double[] weights){
        this.userWeights = weights;
    }
    
    public void updateUserWeights(int weightIndex, double value){
        userWeights[weightIndex] = value;
    }
    
    public double[] getUserWeights(){
        return userWeights;
    }
    
}
