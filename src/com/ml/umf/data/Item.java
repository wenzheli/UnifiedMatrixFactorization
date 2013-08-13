package com.ml.umf.data;

import java.util.Random;

/**
 * Item object for unified matrix factorization. 
 * 
 * @author wenzhe
 *
 */
public class Item extends Entity{
    private double[] latentFeatures;
    private double[] itemFeatures;
    
    // TODO need to set this in config. 
    public static final int numOfUserFeatures = 100;
    public static final int numOfItemFeatures = 100;
    public static final int numOfLatentFeatures = 20;
    
    public Item(){
        super();
        latentFeatures = new double[numOfLatentFeatures];
        // randomly initialize the latent value
        for (int i = 0; i < numOfLatentFeatures; i++){
            latentFeatures[i] = new Random().nextDouble();
        }
        
        itemFeatures = new double[numOfItemFeatures];
    }
    
    public double[] getLatentFeatures(){
        return latentFeatures;
    }
    
    public void setLatentFeatures(double[] latentFeatures){
        this.latentFeatures = latentFeatures;
    }
    
    public double[] getItemFeatures(){
        return itemFeatures;
    }
    
    public void setItemFeatures(double[] itemFeatures){
        this.itemFeatures = itemFeatures;
    }
}
