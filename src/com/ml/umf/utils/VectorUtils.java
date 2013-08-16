package com.ml.umf.utils;

import com.ml.umf.exception.FeatureException;

/**
 * Utillity class for feature vector. Includes inner product, addition, subjection,
 * multiplication..etc. 
 * 
 * @author wenzhe
 *
 */
public class VectorUtils {
    
    /**
     * Inner product of two vectors with the same size. 
     */
    public static double calInnerProduct(double[] feature1, double[] feature2) 
            throws FeatureException{
        double product = 0;
        if (feature1.length != feature2.length){
            throw new FeatureException("Two features have different dimension!");
        }
        
        for (int i = 0; i < feature1.length; i++){
            product += feature1[i] * feature2[i];
        }
        
        return product;
    }
    
    /**
     * Addition of two feature vectors with the same size. 
     */
    public static double[] calAddition(double[] feature1, double[] feature2) 
            throws FeatureException{
        double[] addition = new double[feature1.length];
        if (feature1.length != feature2.length){
            throw new FeatureException("Two features have different dimension!");
        }
        
        for (int i = 0; i < feature1.length; i++){
            addition[i] = feature1[i] + feature2[i];
        }
        
        return addition;
    }
    
    /**
     * Multiply feature vector with a constant. 
     */
    public static double[] calMultiply(double multiplier, double[] feature){
        double[] newFeature = new double[feature.length];
        for (int i = 0; i < feature.length; i++){
            newFeature[i] = multiplier * feature[i];
        }
        
        return newFeature;
    }
    
    /**
     * Calculate the difference between two feature vectors with the same size. 
     */
    public static double[] calMinus(double[] feature1, double[] feature2) 
            throws FeatureException{
        double[] newFeature = new double[feature1.length];
        if (feature1.length != feature2.length){
            throw new FeatureException("Two features have different dimension!");
        }
        for (int i = 0; i < feature1.length; i++){
            newFeature[i] = feature1[i] - feature2[i];
        }
        return newFeature;
        
    }
    
    /**
     * Append two feature vectors. 
     */
    public static double[] appendFeatureVector(double[] feature1, double[] feature2){
        double[] newFeature = new double[feature1.length + feature2.length];
        for (int i = 0; i < feature1.length; i++){
            newFeature[i] = feature1[i];
        }
        for (int i = 0; i < feature2.length; i++){
            newFeature[i + feature1.length] = feature2[i];
        }
        
        return newFeature;
    }
    

    public static double innerProduct(double[] vec1, double[][] weights, double[] vec2){
        double sum = 0;
        for (int i = 0; i < vec1.length; i++){
            for (int j = 0; j < vec2.length; j++){
                sum += vec1[i] * vec2[j] * weights[j][i];
            }
        }
        
        return sum;
    }
}
