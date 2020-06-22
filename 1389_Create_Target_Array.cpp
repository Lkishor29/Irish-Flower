class Solution {
public:
    vector<int> createTargetArray(vector<int>& nums, vector<int>& index) {
        int n=nums.size();
        vector<int>v;
        int i;
        for(i=0;i<nums.size();i++)
        {
            int d=index[i];
            v.insert(v.begin()+d,nums[i]);
        }
        return v;
            
        
    }
};
