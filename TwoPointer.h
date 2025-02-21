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
    // ======= Leetcode Solutions =======

};



extern DLL_API TwoPointer twoPointerInstance;
