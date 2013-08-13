package com.ml.umf.data;

public class UserItem extends Instance{
    
    private double rating;
    
    public UserItem(int userIndex, int itemIndex, double rating){
        super(userIndex, itemIndex);
        this.rating = rating;
    }
    
    public double getRating(){
        return rating;
    }
    
    
}
