package com.ml.umf.data;

/**
 * Our observation, for SGD, we update parameter based on this
 * observation instance. 
 * 
 * @author wenzhe
 *
 */
public abstract class Instance {
    protected int userIndex;
    protected int itemIndex;
    
    public Instance(int userIndex, int itemIndex){
        this.userIndex = userIndex;
        this.itemIndex = itemIndex;
    }
    
    public int getUserIndex(){
        return userIndex;
    }
    
    public int getItemIndex(){
        return itemIndex;
    }
}