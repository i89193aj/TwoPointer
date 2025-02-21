﻿/*Oscar TwoPointer*/

#include <iostream>
#include <cstdlib>  // rand(), srand()
#include <ctime>    // time()
#include <random>   // C++11 亂數庫
#include <vector>   // std::vector
#include <numeric>  // std::iota
#include <map>      //std::map
#include <unordered_map>  //std::unordered_map

#include"TwoPointer.h"
using namespace std;

DLL_API TwoPointer twoPointerInstance;
enum LeetcodeExam {
    Leetcode167TwoSumIIInputArrayIsSorted,

    None,
};

int main()
{
    LeetcodeExam ExamEnum = Leetcode167TwoSumIIInputArrayIsSorted;//ChangeForExam
    vector<int> LinkedlistInput1 = { 7,13,11,10,1 };              //ChangeForExam
    vector<int> LinkedlistInput2 = { 7,13,11,10,1 };              //ChangeForExam
    int iInput1 = 0;int iInput2 = 0;
    int Ans = 0; vector<int> AnsVector;  

    TwoPointer* Implementation = new TwoPointer();

    switch (ExamEnum)
    {
    case Leetcode167TwoSumIIInputArrayIsSorted:
        AnsVector = Implementation->Leetcode_Sol_167(LinkedlistInput1, iInput1,1);
        break;

    default:
        break;
    }
    #pragma region TwoPointer
    TwoPointer obj1;              // 呼叫預設建構式
    obj1.display();

    TwoPointer obj2(10);          // 呼叫帶參數建構式
    obj2.display();

    obj1.setData(20);          // 修改資料成員
    obj1.display();

    return 0;
    #pragma endregion

    
}

#pragma region TwoPointer
// 預設建構式，初始化指標
TwoPointer::TwoPointer() : data(new int(0)) {
    std::cout << "Default constructor called. Data initialized to 0." << std::endl;
}

// 帶參數建構式，初始化指標並設置初始值
TwoPointer::TwoPointer(int value) : data(new int(value)) {
    std::cout << "Parameterized constructor called. Data initialized to " << value << "." << std::endl;
}

// 解構式，釋放動態分配的記憶體
TwoPointer::~TwoPointer() {
    delete data;
    std::cout << "Destructor called. Memory for data released." << std::endl;
}

// 設定資料成員
void TwoPointer::setData(int value) {
    *data = value;
}

// 取得資料成員
int TwoPointer::getData() const {
    return *data;
}

// 顯示資料
void TwoPointer::display() const {
    std::cout << "Data: " << *data << std::endl;
}
#pragma endregion




#pragma region Leetcode 167. Two Sum II - Input Array Is Sorted
//Leetcode 167. Two Sum II - Input Array Is Sorted
vector<int> TwoPointer::Leetcode_Sol_167(vector<int>& numbers, int target,int _solution) {
    switch (_solution)
    {
    case 1:
        return twoSum(numbers, target);
    default:
        return std::vector<int>{}; // 確保所有路徑都有回傳值
    }

    return{};
}

vector<int> TwoPointer::twoSum(vector<int>& numbers, int target) {
    return {};
}
#pragma endregion


