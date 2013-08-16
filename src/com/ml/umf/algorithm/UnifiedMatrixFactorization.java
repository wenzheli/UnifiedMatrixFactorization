package com.ml.umf.algorithm;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import com.ml.umf.common.Classifier;
import com.ml.umf.common.SGDLearner;
import com.ml.umf.data.Instance;
import com.ml.umf.data.Item;
import com.ml.umf.data.User;
import com.ml.umf.data.UserItem;
import com.ml.umf.utils.VectorUtils;

/**
 * Unified matrix factorization. We both use latent features and side side information(
 * item features, user features), in order to solve "cold start" problem. 
 * 
 * @author wenzhe
 *
 */
public class UnifiedMatrixFactorization implements SGDLearner, Classifier{
    
    // free-parameters to control to model, should be learned from cross validation
    // TODO need to set it in config. 
    public double stepSize = 0.01;
    public double regularizationRate = 1;
    public static final int numOfUserFeatures = 100;
    public static final int numOfItemFeatures = 100;
    
    public double[] globalWeights;
    
    public double squareLoss = 0;
    
    /** Item id (index) to Item object map */
    Map<Integer, Item> items = new HashMap<Integer, Item>();
    /** User id (index) to User object map */
    Map<Integer, User> users = new HashMap<Integer, User>();
    
    public UnifiedMatrixFactorization(){
        globalWeights = new double[numOfUserFeatures + numOfItemFeatures];
        // randomly initialize the weights
        for (int i = 0; i < globalWeights.length; i++){
            globalWeights[i] = new Random().nextDouble();
        }
    }
    
    @Override
    public double predict(Instance instance)  {
        // TODO Auto-generated method stub
        int userIndex = ((UserItem)instance).getUserIndex();
        int itemIndex = ((UserItem)instance).getItemIndex();
        User user = users.get(userIndex);
        Item item = items.get(itemIndex);
        double predictedValue = 0;
        try {
            double latentProduct = VectorUtils.calInnerProduct(user.getLatentFeatures(), 
                    item.getLatentFeatures());
            double[] weights = VectorUtils.calAddition(globalWeights, user.getUserWeights());
            double[] userItemFeature = VectorUtils.appendFeatureVector(user.getUserFeatures(),
                    item.getItemFeatures());
            predictedValue = latentProduct + VectorUtils.calInnerProduct(weights, userItemFeature);
        } catch(Exception e){
            // TODO
        }
        return predictedValue;
    }

    @Override
    public void update(Instance instance) throws Exception {
        // TODO Auto-generated method stub
        int userIndex = ((UserItem)instance).getUserIndex();
        int itemIndex = ((UserItem)instance).getItemIndex();
        double rating = ((UserItem)instance).getRating();
        
        User user = users.get(userIndex);
        Item item = items.get(itemIndex);
        
        // update the weights. 
        double latentProduct = VectorUtils.calInnerProduct(user.getLatentFeatures(), 
                item.getLatentFeatures());
        double[] weights = VectorUtils.calAddition(globalWeights, user.getUserWeights());
        double[] userItemFeature = VectorUtils.appendFeatureVector(user.getUserFeatures(),
                item.getItemFeatures());
        double epsilon = latentProduct + VectorUtils.calInnerProduct(weights, userItemFeature) - rating;
        squareLoss += epsilon * epsilon;
        
        // update UserUMF latent feature vector. 
        // u(t) = (1-\eta * lamda)u(t-1)-\eta * epsilon * v(t-1). 
        // u(t) is the UserMF for time t,  and v(t) is ItemMF for time t. 
        double[] firstTerm = VectorUtils.calMultiply(1-stepSize*regularizationRate, user.getLatentFeatures()); 
        double[] secondTerm = VectorUtils.calMultiply(stepSize * epsilon, item.getLatentFeatures()) ;
        double[] updatedUserFeatures = VectorUtils.calMinus(firstTerm, secondTerm);
        
        // update ItemMF latent feature vector
        // v(t) = (1-\eta * lamda)v(t-1) - \eta * epsilon * u(t-1)
        firstTerm = VectorUtils.calMultiply(1-stepSize*regularizationRate, item.getLatentFeatures());
        secondTerm = VectorUtils.calMultiply(stepSize * epsilon, user.getLatentFeatures());
        double[] updatedItemMFFeatures = VectorUtils.calMinus(firstTerm, secondTerm);
        
        // update global weights
        firstTerm = VectorUtils.calMultiply(1-stepSize*regularizationRate, globalWeights);
        secondTerm = VectorUtils.calMultiply(stepSize * epsilon, userItemFeature);
        double[] updatedGlobalWeights = VectorUtils.calMinus(firstTerm, secondTerm);
        
        // update user weights
        firstTerm = VectorUtils.calMultiply(1-stepSize*regularizationRate, user.getUserWeights());
        secondTerm = VectorUtils.calMultiply(stepSize * epsilon, userItemFeature);
        double[] updatedUserWeights = VectorUtils.calMinus(firstTerm, secondTerm);
        
       
        // update each one. 
        user.setLatentFeatures(updatedUserFeatures);
        item.setLatentFeatures(updatedItemMFFeatures);
        globalWeights = updatedGlobalWeights;
        user.setUserWeights(updatedUserWeights);
         
        users.put(userIndex, user);
        items.put(itemIndex, item);
    }
    
}
