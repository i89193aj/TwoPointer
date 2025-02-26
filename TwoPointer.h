#pragma once

#define Build_DLL

#ifdef Build_DLL 
#define DLL_API _declspec(dllexport)
#else 
#define DLL_API _declspec(dllimport)
#endif // BuildDLL _declspec(DLLExport)

#include <string>
#include <iostream>
#include <vector>       // 若成員變數用到 std::vector
#include <map>          // 若有 std::map 變數
#include <unordered_map>// 若有 std::unordered_map 變數

struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
    
};

class TwoPointer {
private:
    int* data;  // 假設資料成員是指標

public:
    // 預設建構式
    TwoPointer();

    // 帶參數建構式
    TwoPointer(int value);

    // 解構式
    ~TwoPointer();

    // 方法：設定值
    void setData(int value);

    // 方法：取得值
    int getData() const;

    // 顯示資料
    void display() const;

    // ======= Leetcode Solutions =======
    std::vector<int> Leetcode_Sol_167(std::vector<int>& numbers, int target, int _solution);
    std::vector<int> Map_167(std::vector<int>& numbers, int target);
    std::vector<int> TwoPointer_167(std::vector<int>& numbers, int target);

    std::vector<int> Leetcode_Sol_658(std::vector<int>& numbers, int k, int x, int _solution);
    std::vector<int> TwoPointer_658(std::vector<int>& numbers, int k, int x);
    std::vector<int> BinarySearchAndTwoPointer_658(std::vector<int>& numbers, int k, int x);

    ListNode* Leetcode_Sol_19(ListNode* head, int n, int _solution);
    ListNode* TwoPointer_19(ListNode* head, int n);
    ListNode* OnePointer_19(ListNode* head, int n);

    std::vector<std::vector<int>> Leetcode_Sol_15(std::vector<int>& nums, int _solution);
    std::vector<std::vector<int>> TwoPointer_15(std::vector<int>& nums);
    void Sort_15(std::vector<int>& nums, int first, int last);
    void OutPlace_15(std::vector<int>& nums, int first,int middle, int last);

    std::vector<std::vector<int>> Leetcode_Sol_18(std::vector<int>& nums,int target, int _solution);
    std::vector<std::vector<int>> TwoPointer_18(std::vector<int>& nums, int target);
    std::vector<std::vector<int>> Universal_18(std::vector<int>& nums, int target);
    void nSum(const std::vector<int>& nums, long n, long target, int l, int r, std::vector<int>& path, std::vector<std::vector<int>>& ans);

    void Leetcode_Sol_75(std::vector<int>& nums, int _solution);
    void MergeSort_75(std::vector<int>& nums, int first, int last);
    void Merge_75(std::vector<int>& nums, int first, int middle, int last);
    void QuickSort_75(std::vector<int>& nums, int first, int last);
    int getpivot(std::vector<int>& nums, int first, int last);
    void Counting_75(std::vector<int>& nums);
    void ThreePointer_75(std::vector<int>& nums);

    void Leetcode_Sol_88(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n);

    std::string Leetcode_Sol_5(std::string s, int _solution);
    std::string TwoPointerOfExpandAroundCenter_5(std::string s);
    std::string ManachersAlg_5(std::string s);




    // ======= Leetcode Solutions =======

};



extern DLL_API TwoPointer twoPointerInstance;
