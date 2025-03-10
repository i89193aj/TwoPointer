﻿/*Oscar TwoPointer*/

#include <iostream>
#include <cstdlib>  // rand(), srand()
#include <ctime>    // time()
#include <random>   // C++11 亂數庫
#include <vector>   // std::vector
#include <numeric>  // std::iota
#include <map>      //std::map
#include <unordered_map>  //std::unordered_map
#include <deque>    //std::deque
#include<algorithm> //std::lower_bound、std::upper_bound(如果不加這行，其他編譯器 (如 g++, clang++) 可能會報錯)
#include <ranges>   // for ranges

#include"TwoPointer.h"
using namespace std;

DLL_API TwoPointer twoPointerInstance;
enum LeetcodeExam {
    Leetcode167TwoSumIIInputArrayIsSorted,
    Longest5PalindromicSubstring,
    None,
};

int main()
{
    LeetcodeExam ExamEnum = Longest5PalindromicSubstring;         //ChangeForExam
    vector<int> LinkedlistInput1 = { 7,13,11,10,1 };              //ChangeForExam
    vector<int> LinkedlistInput2 = { 7,13,11,10,1 };              //ChangeForExam
    int iInput1 = 0;int iInput2 = 0;
    //string strinput1 = "gggggacdbabbbbbbbbbbbbbbabdcabbbbbbbbbbbbbbabdcazzzz";
    string strinput1 = "bab";
    string strinput2 = "xaabacxcabaaxcabaax";

    int Ans = 0; vector<int> AnsVector; string AnsStr = "";

    TwoPointer* Implementation = new TwoPointer();

    switch (ExamEnum)
    {
    case Leetcode167TwoSumIIInputArrayIsSorted:
        AnsVector = Implementation->Leetcode_Sol_167(LinkedlistInput1, iInput1,1);
        break;
    case Longest5PalindromicSubstring:
        AnsStr = Implementation->Leetcode_Sol_5(strinput1,2);
        AnsStr = Implementation->Leetcode_Sol_5(strinput2, 2);
        AnsStr = Implementation->Leetcode_Sol_5(strinput2, 3);
        break;
    default:
        break;
    }

    #pragma region vector
        std::vector<int> nums_test;
        nums_test.reserve(4);  // 預留 4 個空間

        // 嘗試訪問 nums[4]
        //std::cout << nums[4] << std::endl;  // 這裡會導致未定義行為
    #pragma endregion

    
    #pragma region TwoPointer
    TwoPointer obj1;              // 呼叫預設建構式
    obj1.display();

    TwoPointer obj2(10);          // 呼叫帶參數建構式
    obj2.display();

    obj1.setData(20);          // 修改資料成員
    obj1.display();

    return 0;
    #pragma endregion

    #pragma region 16進位, 2進位運算
    /*16進位1位數 = 4bit(0000 ~ 1111)，超過1111就進位，所以1byte就等於16進制+1*/
    int arr[] = { 10, 20, 30, 40 };

    int* p1 = &arr[1]; // 20
    int* p2 = &arr[2]; // 30

    cout << "p1: " << p1 << endl;
    cout << "p2: " << p2 << endl;
    cout << "p1 < p2: " << (p1 < p2) << endl;
    cout << "p1 < p2: " << (p1 > p2) << endl;

    int arr1[] = { 40,30,20,10 };
    p1 = &arr1[1]; // 30
    p2 = &arr1[2]; // 20
    cout << "p1: " << p1 << endl;
    cout << "p2: " << p2 << endl;
    cout << "p1 < p2: " << (p1 < p2) << endl;
    cout << "p1 < p2: " << (p1 > p2) << endl;
    #pragma endregion


    
}

#pragma region Class:TwoPointer
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

#pragma region 考題一:左右指針(相向)：1.排序數組 2.回文 3.最大面積
#pragma region Leetcode 11. Container With Most Water
//Leetcode 11. Container With Most Water
int TwoPointer::Leetcode_Sol_11(vector<int>& height) {
    /*左右指針*/
    int maxwater = 0, max_temp = 0;
    int left = 0, right = height.size() - 1;
    int height_max = 0;
    while (left < right) {
        if (height[right] > height[left]) {
            max_temp = height[left] * (right - left);
            left++;
        }
        else {
            max_temp = height[right] * (right - left);
            right--;
        }
        maxwater = maxwater > max_temp ? maxwater : max_temp;
    }
    return maxwater;
}
#pragma endregion

#pragma region Leetcode 125. Valid Palindrome
//Leetcode 125. Valid Palindrome
bool TwoPointer::Leetcode_Sol_125(string s) {
    /*左右指針*/
    int left = 0, right = s.size() - 1;
    while (left <= right) {
        if (!isalnum(s[left])) left++;
        else if (!isalnum(s[right])) right--;
        else if (tolower(s[left++]) != tolower(s[right--])) return false;
    }
    return true;
}
#pragma endregion

#pragma region Leetcode 167. Two Sum II - Input Array Is Sorted
//Leetcode 167. Two Sum II - Input Array Is Sorted
vector<int> TwoPointer::Leetcode_Sol_167(vector<int>& numbers, int target,int _solution) {
    switch (_solution)
    {
    case 1:
        return Map_167(numbers, target);
    case 2:
        return TwoPointer_167(numbers, target);
    default:
        return std::vector<int>{}; // 確保所有路徑都有回傳值
    }

    return{};
}

vector<int> TwoPointer::Map_167(vector<int>& numbers, int target) {
    /*
         num1 + num2 = target
         num1 = target - num2
         unordered_map[num2] = num1 => i;
    */
    unordered_map<int, int> map;
    for (int i = 0;i < numbers.size();i++) {
        if (map.find(numbers[i]) == map.end())
            map[target - numbers[i]] = i;
        else
            return { map[numbers[i]] ,i};
    }

    return {};
}
/*左右指針（Opposite Direction Two Pointers）*/
vector<int> TwoPointer::TwoPointer_167(vector<int>& numbers, int target) {
    /*
    * Because it's in non-decreasing sort, so it's can use TwoPointer to Solution
    * Trap：added by one =>Original：Return the indices of the two numbers, index1 and index2, "added by one" as an integer array [index1, index2] of length 2.
    */
    int p1 = 0; int p2 = numbers.size() - 1;
    while (p1 < p2) {
        if (numbers[p1] + numbers[p2] == target) return { ++p1,++p2 };
        else if (target - numbers[p1] < numbers[p2])p2--;
        else if (target - numbers[p2] > numbers[p1])p1++;
        else
        {
            p1++; p2--;
        }
    }
    return {};
}
#pragma endregion

#pragma region Leetcode 15. 3Sum
//Leetcode 15. 3Sum(排序數組)
vector<vector<int>> TwoPointer::Leetcode_Sol_15(vector<int>& nums, int _solution) {
    switch (_solution)
    {
    case 1:
        return TwoPointer_15(nums);
    case 2:
        return TwoPointerByiterator_15(nums);
    default:
        return {}; // 確保所有路徑都有回傳值
    }

    return {};
}
/*雙指針 + 排序	KSum 問題=> O(n^2)*/
vector<vector<int>> TwoPointer::TwoPointer_15(vector<int>& nums) {
    //Sort_15(nums,0,nums.size()-1);    //自己寫的mergesort
    std::sort(nums.begin(), nums.end()); //std的sort
    vector<vector<int>> ans; int size = nums.size();
    ans.reserve(size / 3); //直接預留空間，不然超過會一直重新創，浪費時間
    for (int i = 0; i < size - 2/*左右各一*/; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;         //第二次打還是會忘記重複的直接省去
        int l = i + 1, r = size - 1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                ans.push_back({ nums[i], nums[l], nums[r] }); \
                    l++;
                r--;
                while (l < r && nums[l] == nums[l - 1]) l++;    //重複的就別錯了，EX：[1,1......105]
                while (l < r && nums[r] == nums[r + 1])r--;     //第二次打還是會忘記重複的直接省去
            }
            else if (sum < 0) l++;
            else r--;
        }
    }
    return ans;
}

vector<vector<int>> TwoPointer::TwoPointerByiterator_15(vector<int>& nums) {
    sort(nums.begin(), nums.end());

    vector<vector<int>> out;
    //for(int i = 0; i < nums.size() && nums[i] <= 0; ++i){
    for (auto it = nums.begin(); it != nums.end() && *it <= 0; ++it) {
        //if(i && nums[i] == nums[i-1]) continue; // skip duplicates
        if (it != nums.begin() && *it == *(it - 1)) continue; // skip duplicates

        //int left = i + 1;
        auto left = it + 1;
        //int right = nums.size()-1;
        auto right = nums.end() - 1;

        while (left < right) {
            //int sum = nums[i] + nums[left] + nums[right];
            int sum = *it + *left + *right;
            if (sum > 0) { // too big, shrink largest number
                right--;
            }
            else if (sum < 0) { //too small, grow smallest number
                left++;
            }
            else { //matched
                //out.push_back({nums[i], nums[left], nums[right]});
                out.push_back({ *it, *left, *right });
                //Consume pair
                left++;
                right--;
                // skip duplicates
                for (; left < right && *left == *(left - 1); ++left) {}
            }
        }
    }

    return out;
}

void TwoPointer::Sort_15(vector<int>& nums, int first, int last) {

    if (last <= first) return;
    int middle = (first + last) >> 1;
    Sort_15(nums, first, middle);
    Sort_15(nums, middle + 1, last);
    OutPlace_15(nums, first, middle, last);

}

void TwoPointer::OutPlace_15(vector<int>& nums, int first, int middle, int last) {
    int iSortsize = last - first + 1; vector<int> sort(iSortsize, 0); int i = 0;
    int l = first; int r = middle + 1;
    while (l <= middle && r <= last)
        sort[i++] = nums[l] < nums[r] ? nums[l++] : nums[r++];
    while (l <= middle)
        sort[i++] = nums[l++];
    while (r <= last)
        sort[i++] = nums[r++];

    for (int i = 0; i < iSortsize; i++)
        nums[first + i] = sort[i];
}
#pragma endregion

#pragma region Leetcode 18. 4Sum
//Leetcode 18. 4Sum (排序數組)
/*雙指針 + 排序	KSum 問題=> O(n^3)*/
vector<vector<int>> TwoPointer::Leetcode_Sol_18(vector<int>& nums, int target, int _solution) {
    switch (_solution)
    {
    case 1:
        return TwoPointer_18(nums, target);
    case 2:
        return TwoPointeriterator_18(nums, target);
    case 3:
        return Universal_18(nums, target);
    default:
        return {}; // 確保所有路徑都有回傳值
    }

    return {};
}

vector<vector<int>> TwoPointer::TwoPointer_18(vector<int>& nums, int target) {
    //Sort_15(nums,0,nums.size()-1);    //自己寫的mergesort
    std::sort(nums.begin(), nums.end()); //std的sort
    vector<vector<int>> ans; int size = nums.size();
    int sizeof3sum = size - 2, sizeof4sum = size - 3;
    for (int i_4 = 0; i_4 < sizeof4sum; i_4++) {
        if (i_4 > 0 && nums[i_4] == nums[i_4 - 1]) continue;

        for (int i_3 = i_4 + 1; i_3 < sizeof3sum; i_3++) {    //錯誤：i_3應該在i_4後面
            if (i_3 > i_4 + 1 && nums[i_3] == nums[i_3 - 1]) continue;//錯誤：if (nums[i_3] == nums[i_3 - 1]) continue;

            int l = i_3 + 1, r = size - 1;
            while (l < r) {
                long sum = nums[i_4];
                sum += nums[i_3];
                sum += nums[l];
                sum += nums[r];
                //錯誤：long long sum = nums[i_4] + nums[i_3] + nums[l] + nums[r];
                /*這題超陷阱，因為int加全部的時候，都是在int加的，他是加完後在轉long，除非你先個別轉，如下：
                long sum = static_cast<long>(nums[i_4]) + static_cast<long>(nums[i_3]) + static_cast<long>(nums[l]) + static_cast<long>(nums[r]);*/

                if (sum == target) {
                    ans.push_back({ nums[i_4] ,nums[i_3] ,nums[l] ,nums[r] }); //記得排序好
                    l++; r--;
                    while (l < r && nums[l] == nums[l - 1]) l++;
                    while (l < r && nums[r] == nums[r + 1]) r--;
                }
                else if (sum < target)l++;
                else r--;
            }
        }
    }
    return ans;
}

vector<vector<int>> TwoPointer::TwoPointeriterator_18(vector<int>& nums, int target) {
    // 避免越界(如果你下面那個判斷式要寫成，it!=nums.end()-3、it!=nums.end()-2的形式)那就要加這行
    //if (nums.size() < 4) return {}; 
    sort(nums.begin(), nums.end());
    vector<vector<int>> ans;
    ans.reserve(nums.size() / 4);
    for (auto it = nums.begin(); it != nums.end(); ++it) {
        if (it != nums.begin() && *it == *(it - 1)) continue;
        for (auto it2 = it + 1; it2 != nums.end(); ++it2) {
            if (it2 != it + 1 && *it2 == *(it2 - 1)) continue;

            vector<int>::iterator l = it2 + 1, r = nums.end() - 1;
            while (l < r) {
                long long sum = (long long)*it + *it2 + *l + *r;
                if (sum > target) r--;
                else if (sum < target)l++;
                else {
                    ans.push_back({ *it,*it2,*l++,*r-- });
                    //錯誤寫法：ans.emplace_back(*it, *it2, *l, *r);
                    /*正確寫法：
                    ans.emplace_back(std::vector<int>{*it, *it2, *l, *r});
                    // 直接傳入參數，讓 emplace_back 建構 vector<int>
                    l++;
                    r--;*/
                    while (l < r && *l == *(l - 1)) l++;
                    while (l < r && *r == *(r + 1)) r--;
                }
            }
        }
    }
    return ans;
}

/*星星：***** 通用nSum */
vector<vector<int>> TwoPointer::Universal_18(vector<int>& nums, int target) {
    vector<vector<int>> ans; vector<int> path;
    sort(nums.begin(), nums.end());
    nSum(nums, 4, target, 0, nums.size() - 1, path, ans);//第二個參數：控制第nSum
    return ans;
}

void TwoPointer::nSum(const vector<int>& nums, long n, long target, int l, int r, vector<int>& path, vector<vector<int>>& ans) {
    //Base Case Condition：
    if (r - l + 1 < n || target < n * nums[l] || target > n * nums[r]) return;//長度要對 || 目標值都比剩下的還小 || 目標值都比剩下的還大
    //Base Case：
    if (n == 2) {
        while (l < r) {
            int sum = nums[l] + nums[r];//不知道為什麼作者要寫const
            if (sum == target) {
                path.push_back(nums[l]);
                path.push_back(nums[r]);
                ans.push_back(path);
                path.pop_back();
                path.pop_back();
                r--; l++;
                while (l < r && nums[l] == nums[l - 1]) l++;
                while (l < r && nums[r] == nums[r + 1]) r--;
            }
            else if (target > sum)l++;
            else r--;
        }
        return;
    }
    //Recursion
    int iChangeSize = r - (n - 2);
    for (int i = l; i <= iChangeSize; i++) {
        if (i > l && nums[i] == nums[i - 1]) continue;
        path.push_back(nums[i]);                       //為了帶上一層的參數
        nSum(nums, n - 1, target - nums[i], i + 1, r, path, ans);//遞迴n-1的nSum
        path.pop_back();                               //用完後，要pop()
    }
}
#pragma endregion

#pragma region Leetcode 658. Find K Closest Elements
//Leetcode 658. Find K Closest Elements (也是考排序 + 某一值的左右)
vector<int> TwoPointer::Leetcode_Sol_658(vector<int>& numbers, int k, int x, int _solution) {
    switch (_solution)
    {
    case 1:
        return TwoPointer_658(numbers, k, x);
    case 2:
        return BinarySearchAndTwoPointer_658(numbers, k, x);
    default:
        return vector<int>{}; // 確保所有路徑都有回傳值
    }

    return{};
}

vector<int> TwoPointer::TwoPointer_658(vector<int>& numbers, int k, int x) {
    int left = 0; int right = 0; deque<int> anstemp;//記住錯誤push_back、push_front是額外加入元素並不是取代元素
    int endidx = numbers.size() - 1;
    while (numbers[left] < x && left < endidx) {
        left++;
    }
    left--;
    right = left + 1 >= endidx ? endidx : left + 1;
    int iCount = 0;
    while (iCount != k && left >= 0 && right <= endidx) {//邊界條件設在這裡
        if (x - numbers[left] > numbers[right] - x) { //這個不能left == 0，邊界值不能這樣設，會有剛好的問題
            anstemp.push_back(numbers[right]);
            right++;
        }
        else if (x - numbers[left] <= numbers[right] - x) {
            anstemp.push_front(numbers[left]);
            left--;
        }

        iCount++;
    }

    while (iCount != k && left >= 0) {
        anstemp.push_front(numbers[left]);
        left--;
        iCount++;
    }

    while (iCount != k && right <= endidx) {
        anstemp.push_back(numbers[right]);
        right++;
        iCount++;
    }

    return  vector<int>(anstemp.begin(), anstemp.end());
}
/* 星星：*****
* 1.了解deque怎麼用(包含deque轉vector)，O(1)
* 2.了解binarysearc怎麼用(lower_bound、upper_bound)，O(logn)
* 3.TwoPointer(O(k))
* Time Complexity：(O(k+logn))
*/
vector<int> TwoPointer::BinarySearchAndTwoPointer_658(vector<int>& numbers, int k, int x) {
    int left = lower_bound(numbers.begin(), numbers.end(), x) - numbers.begin() - 1;//O(logn)
    int right = left + 1;
    deque<int> anstemp;
    //O(k)
    while (k--) {
        if (right >= numbers.size() || (left >= 0 && x - numbers[left] <= numbers[right] - x))
            anstemp.push_front(numbers[left--]);
        else
            anstemp.push_back(numbers[right++]);
    }

    return vector<int>(anstemp.begin(), anstemp.end());
}
#pragma endregion

#pragma region Leetcode 42. Trapping Rain Water
//Leetcode 42. Trapping Rain Water(3.最大面積)
int TwoPointer::Leetcode_Sol_42(vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int max_left = 0, max_right = 0;
    int icount = 0;
    while (left <= right) {//判斷要不要等於，就相當於要不要算最後相等的那次
        if (height[left] < height[right]) {
            if (height[left] >= max_left)
                max_left = height[left];
            else
                icount += max_left - height[left];
            left++;
        }
        else {
            if (height[right] >= max_right)
                max_right = height[right];
            else
                icount += max_right - height[right];
            right--;
        }
    }
    return icount;
#pragma region 解題思路:
    /*
          step1. 先拆成U 型 跟 倒U型：
          那自然而已我們會寫成左右向中間走(並且是小的往間走)，
          再來記錄當前left的左邊最大高度、right的右邊最大高度有沒有比自己大或是跟自己一樣大：
          如果>=的話，就刷新當前最大值
          step2. 那如果沒有的話要怎麼紀錄呢?
          就把當前的水位紀錄起來
          (那可能會想說奇怪，那萬一右邊比較矮呢?)
          Ans：原因當你會記錄這次水位的前提 => 右邊就已經比左邊大才會記錄這次left的水位，
               右邊永遠會高於左邊，因為第一層的判別，就是大的不動，小的動；反之也是一樣
     */
#pragma endregion
}
#pragma endregion
#pragma endregion

#pragma region 考題二:同向指針 1.子陣列 2.滑動窗口(Sliding Window)

#pragma region Leetcode 3. Longest Substring Without Repeating Characters
//Leetcode 11. Longest Substring Without Repeating Characters
/*標準slinding window*/
int TwoPointer::Leetcode_Sol_3(string s) {
    int left = 0, right = 0;
    vector<int> freq(128, 0);
    int max_length = 0;
    while (right < s.size()) {
        if (!freq[s[right]])
            freq[s[right++]]++;
        else { //存在
            if (freq[s[left]]) freq[s[left++]]--;
        }
        int maxtemp = right - left;
        max_length = max_length < maxtemp ? maxtemp : max_length;
    }
    return max_length;
}
#pragma endregion

#pragma region Leetcode 209. Minimum Size Subarray Sum
//Leetcode 209. Minimum Size Subarray Sum
/*標準slinding window*/
int TwoPointer::Leetcode_Sol_209(std::vector<int>& nums, int target) {
    int left = 0, right = 0;
    int min_length = INT_MAX;
    int sum = 0;
    while (right < nums.size()) {
        sum += nums[right];
        if (sum < target)
            right++;
        else { //這時候已經>= target
            int lengthtemp = right - left + 1;
            min_length = min_length < lengthtemp ? min_length : lengthtemp;
            sum -= nums[right];
            sum -= nums[left++];
        }
    }
    return min_length == INT_MAX ? 0 : min_length;
}
#pragma endregion

#pragma region Leetcode 76. Minimum Window Substring
//Leetcode 76. Minimum Window Substring(1.子陣列)
string TwoPointer::Leetcode_Sol_76(std::string s, std::string t) {
    int Tsize = t.size(), Ssize = s.size();
    if (Ssize < Tsize) return "";
    //Bucket Counting =>「利用一個固定大小的陣列來統計頻率」
    vector<int> map(128, 0);
    vector<int> freq(128, 0);
    for (auto c : t) {
        freq[c]++;
    }
    int left = 0;   //當第一次滿足所有t的時候，開始跟著跑紀錄位置(跑一次m)
    int right = 0;  //right從頭到尾遍歷一次(跑一次m)
    int count = 0;  //用來確認t的所有值都有出現
    int min_length = INT_MAX, start = 0;
    int freqCount = count_if(freq.begin(), freq.end(), [](int x) {return x > 0; });
    while (right < Ssize) {
        //擴大窗口
        if (freq[s[right]] > 0) {
            map[s[right]]++;
            if (map[s[right]] == freq[s[right]])
                count++;
        }
        //當全部t內的char都有時，收縮窗口
        while (count == freqCount) {
            if (freq[s[left]]) {
                if (right - left + 1 < min_length) {
                    start = left;
                    min_length = right - left + 1;
                }
                /*這行是為了當left遇到第一個t字串裡面的內容時，
                  為了能讓right不管後續有多少個重複的t字串裡面的char，最後都必須由這個收尾*/
                if (map[s[left]] == freq[s[left]]) {
                    count--;//(寫這個是因為等等後面會做map[s[left]]--;減少他的出現次數)
                }
                map[s[left]]--; //會有abcba這種情況
            }

            left++;

        }
        right++;
    }
    return min_length == INT_MAX ? "" : s.substr(start, min_length);
}
#pragma endregion
#pragma endregion

#pragma region 考題三:快慢指針 1.環檢測(one step + two step) 2.中點查找 3.fast先跑,再與slow一起跑

#pragma region Leetcode 141. Linked List Cycle
//Leetcode 141. Linked List Cycle
/*fast slow pointer*/
bool TwoPointer::Leetcode_Sol_141(ListNode* head){
    ListNode* Dummy = new ListNode(-1, head);
    ListNode* fast = Dummy, * slow = Dummy;
    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
        if (fast == slow) return true;
    }
    return false;
}
#pragma endregion

#pragma region Leetcode 287. Find the Duplicate Number
//Leetcode 287. Find the Duplicate Number
/*PigeonHole Principle && Floyd’s Tortoise and Hare (Cycle Detection) && Fast slow Pointer*/
int TwoPointer::Leetcode_Sol_287(vector<int>& nums) {
    int fast = nums[0], slow = nums[0];
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (fast != slow);
    slow = nums[0];
    while (fast != slow) {
        slow = nums[slow];
        fast = nums[fast];
    }
    return fast;
}
#pragma endregion

#pragma region Leetcode 19. Remove Nth Node From End of List
//Leetcode 19. Remove Nth Node From End of List
ListNode* TwoPointer::Leetcode_Sol_19(ListNode* head, int n, int _solution) {
    switch (_solution)
    {
    case 1:
        return OnePointer_19(head, n);
    case 2:
        return TwoPointer_19(head, n);
    default:
        return nullptr; // 確保所有路徑都有回傳值
    }

    return nullptr;
}

ListNode* TwoPointer::OnePointer_19(ListNode* head, int n) {
    ListNode* Dummy = new ListNode(-1, head);
    ListNode* fast = head; int idx = 0; int iCount = 0;
    while (fast) {
        fast = fast->next;
        iCount++;
    }
    int iShift = iCount - n;
    fast = Dummy;
    while (idx != iShift) {
        fast = fast->next;
        idx++;
    }
    ListNode* delnode = fast->next;
    fast->next = fast->next->next;
    delete delnode;
    return  Dummy->next;
}

ListNode* TwoPointer::TwoPointer_19(ListNode* head, int n) {
    ListNode* Dummy = new ListNode(-1, head);// 重要：虛擬頭節點，避免刪除第一個節點的特判
    ListNode* fast = Dummy;
    ListNode* slow = Dummy;

    //我們需要知道被截掉的前一段，所以差距需要+1
    for (int i = 0; i <= n/*i < n + 1*/; i++)
        fast = fast->next;

    //因為她是往回刪除，所以利用最後的間距(把這個間距平移到最前面 =>相當於間距的平移)
    while (fast) {
        fast = fast->next;
        slow = slow->next;
    }
    //這時候的slow是他的前一個節點(刪除 slow->next（即倒數第 N 個節點）)
    ListNode* nodeToDelete = slow->next;
    slow->next = slow->next->next;
    delete nodeToDelete;

    return Dummy->next;
}
#pragma endregion
#pragma endregion

#pragma region 考題四:ThreePointer

#pragma region Leetcode 75. Sort Colors
//Leetcode 75. Sort Colors
/*Dutch National Flag Algorithm (DNF) => ThreePointer*/
void TwoPointer::Leetcode_Sol_75(vector<int>& nums, int _solution) {
    int last = nums.size() - 1, first = 0;
    switch (_solution)
    {
    case 1:
        return MergeSort_75(nums,first,last);
    case 2:     
        return QuickSort_75(nums, first, last);       
    case 3:
        return Counting_75(nums);
    case 4:
        return ThreePointer_75(nums);

    default:
        return; // 確保所有路徑都有回傳值
    }

    return;
}

void TwoPointer::MergeSort_75(vector<int>& nums,int first,int last) {
    if (first >= last) return;
    int middle = (last + first) >> 1;
    MergeSort_75(nums,first, middle);
    MergeSort_75(nums, middle+1,last);
    Merge_75(nums,first, middle,last);
}
void TwoPointer::Merge_75(vector<int>& nums, int first,int middle, int last) {
    int sortsize = last - first + 1; vector<int> sort(sortsize,0);
    int step = 0, l = first, r = middle + 1;
    while (l <= middle && r <= last)
        sort[step++] = nums[l] < nums[r] ? nums[l++] : nums[r++];       
    while (l <= middle)
        sort[step++] = nums[l++];
    while (r <= last)
        sort[step++] = nums[r++];

    std::copy(sort.begin(),sort.end(),nums.begin() + first);
}

void TwoPointer::QuickSort_75(vector<int>& nums, int first, int last) {
    if (first >= last) return;  
    swap(nums[first + rand() % (last - first + 1)],nums[first]);
    int pivot = getpivot(nums,first,last);

    QuickSort_75(nums,first, pivot);
    QuickSort_75(nums,pivot+1,last);
}

int TwoPointer::getpivot(vector<int>& nums, int first, int last) {
    int pivot = first; int l = pivot + 1, r = last;
    while (l <= r) {
        if (nums[l] < nums[pivot])l++;
        else if (nums[r] >= nums[pivot]) r--;
        else swap(nums[l],nums[r]);
    }
    swap(nums[pivot],nums[r]);
    return r;
}

void TwoPointer::Counting_75(vector<int>& nums) {
#pragma region 太大不適合這題
    //auto result = minmax_element(nums.begin(), nums.end());
   //int max = *result.first, min = *result.second;
   //int feqsize = max - min + 1;  vector<int> feq(feqsize, 0);

   ////出現頻率
   //for (auto num : nums)
   //    feq[num - min]++;//錯誤：feq[num]++; 記得是-min
   ////累加頻率(紀錄位置)
   //for (int i = 1; i < feqsize; i++)
   //    feq[i] += feq[i - 1];

   ////排序
   //vector<int> sort(nums.size(), 0);
   //for (int i = nums.size() - 1; i >= 0; i--)  //錯誤：已經很多次了!!!，後面那個是i--，不是i++
   //    sort[feq[nums[i] - min]-- - 1] = nums[i];//錯誤：sort[feq[nums[i]]-- -1] = nums[i]; 這裡記得也要-min

   //std::copy(sort.begin(), sort.end(), nums.begin());
#pragma endregion
    int feqsize = 3;  vector<int> feq(feqsize, 0);

    //出現頻率
    for (auto num : nums)
        feq[num]++;
    //累加頻率(紀錄位置)
    for (int i = 1; i < feqsize; i++)
        feq[i] += feq[i - 1];

    //排序
    vector<int> sort(nums.size(), 0);
    for (int i = nums.size() - 1; i >= 0; i--)
        sort[feq[nums[i]]-- - 1] = nums[i];

    std::copy(sort.begin(), sort.end(), nums.begin());
}

void TwoPointer::ThreePointer_75(vector<int>& nums) {
    int l = 0; int mid = 0; int r = nums.size() - 1;
    while (mid<=r) {
        if (!nums[mid]) swap(nums[mid++], nums[l++]); // 當 nums[mid] 為 0，將其放到最左邊，low 指針向右移動
        else if (nums[mid] == 1) mid++;               // 當 nums[mid] 為 1，表示已經是正確的顏色，mid 直接向右移動
        else swap(nums[mid], nums[r--]);              // 當 nums[mid] 為 2，將其放到最右邊，high 指針向左移動
    }
}

#pragma endregion

#pragma region Leetcode 88. Merge Sorted Array 
//Leetcode 88. Merge Sorted Array 
void TwoPointer::Leetcode_Sol_88(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n) {
    /*Three Pointer*/
    //他說已經排序過
    //sort(nums1.begin(), nums1.begin()+m);//[start, end) 左閉右開區間
    //sort(nums2.begin(), nums2.begin() + n);

    int p1 = m - 1, p2 = n - 1, p3 = n + m - 1;
    while (p1>= 0 && p2 >=0) {
        if (nums1[p1] > nums2[p2]) nums1[p3--] = nums1[p1--];
        else nums1[p3--] = nums2[p2--];
    }
    while (p1 >= 0)
        nums1[p3--] = nums1[p1--];
    while (p2 >= 0)
        nums1[p3--] = nums2[p2--];
}
#pragma endregion
#pragma endregion

#pragma region 考題一(變形題):左右指針(相向) + 回文 => (Manacher面試不可寫，但要會說明)
#pragma region Leetcode 5. Longest Palindromic Substring
//Leetcode 5. Longest Palindromic Substring

string TwoPointer::Leetcode_Sol_5(string s, int _solution) {
    switch (_solution)
    {
    case 1:
        return TwoPointerOfExpandAroundCenter_5(s);
    case 2:
        return ManachersAlg_5(s);
    case 3:
        return ManachersAlg_iterator_5(s);

    default:
        return ""; // 確保所有路徑都有回傳值
    }

    return "";
}

string TwoPointer::TwoPointerOfExpandAroundCenter_5(string s) {
    if (s.empty()) return "";

    int start = 0, maxLen = 0;
    auto expand = [&](int l, int r) {
        while (l >= 0 && r < s.size() && s[l] == s[r]) {
            l--, r++;
        }
        if (maxLen < r - l - 1) {
            maxLen = r - l - 1;
            start = l + 1;
        }
        };

    for (int i = 0; i < s.size(); i++) {
        expand(i, i);     // 奇數長度回文
        expand(i, i + 1); // 偶數長度回文
    }

    return s.substr(start, maxLen);
#pragma region Lambda List
    /*
    Lambda列表：
    [&]	所有變數都用引用捕獲（可以修改外部變數）
    [=]	所有變數都用值捕獲（不能修改外部變數）
    [var]	只捕獲 var，用 值捕獲
    [&var]	只捕獲 var，用 引用捕獲
    [=, &var]	預設值捕獲，但 var 例外 用引用捕獲
    [&, var]	預設引用捕獲，但 var 例外 用值捕獲
    */
#pragma endregion   
}
#pragma region - Memorize1 - 
/*Step 1.
原始："abc" → 轉換後："#a#b#c#"
這樣可以避免奇偶數長度的回文判斷問題。

Step 2.
當前i(整個遍歷一次)進行中心擴展

Step 3.
 r（當前回文右邊界）
 c（當前回文中心）
 (只要下一個回文的右邊界(i+p[i])大於當前r，那r、c就會更新)
 p[i] 為以 i 為中心的回文半徑（包含 #）

 Step 5.(這步通常會在寫完後，再寫，因為他算是邊界條件，設定當前的最小回文半徑)
 鏡像原理加速計算：
 若 i 在 r 內部，則 p[i] 至少等於 p[2c - i]
 （鏡像對稱：因為現在在當前最大回文的左邊，只需要看最大回文的右邊(當前i以c對稱)是否有回文，但如果右邊回文半徑比當前的r還大，那就會用r限制）。
 若 p[i] 擴展超過 r，更新 c 和 r。

 Step 4.
 若當前回文半徑，比之前的最大回文半徑還大，那就會刷新紀錄*/
#pragma endregion

#pragma region - Memorize2 -
 /*
 另一種背法：
 馬拉車算法記憶法：3+1 法則
3 大核心步驟 + 1 個初始化
✅ Step 0: 初始化（預處理）
將字串轉換成 奇偶長度統一的格式（加 #）。
"abc" → "#a#b#c#"
設定：
c = 0（回文中心）
r = 0（回文右邊界）
p[i]（存 i 為中心的回文半徑）
✅ Step 1: 鏡像加速
計算 mirror = 2c - i（對 c 鏡像的對稱點）
若 i < r，則：
p[i] = min(p[mirror], r - i)（但不能超過 r - i）
✅ Step 2: 中心擴展
從 p[i] 的基礎值開始嘗試擴展回文：
若兩邊相等（t[i - p[i]] == t[i + p[i]]），則 p[i]++
直到不匹配為止。
✅ Step 3: 更新中心與右邊界
若 i + p[i] > r（發現更大的回文），則：
更新 c = i
更新 r = i + p[i]*/
#pragma endregion
string TwoPointer::ManachersAlg_5(string s) {
    if (s.empty()) return "";
    //Step 1.
    string t = "#";
    for (char c : s) t += c, t += "#";
    int n = t.size(), center = 0, right = 0, maxLen = 0, start = 0;
    vector<int> p(n, 0);//記錄所有元素的半徑(容易忘)

    for (int i = 0; i < n; i++) {
        //Step 5. 利用鏡像原理：設定當前最小半徑(詳情看上面解說)
        int mirror = 2 * center - i;
        if (i < right) p[i] = min(right - i, p[mirror]);

        //Step 2. 中心擴展(+1, -1式直接看下一個，因為半徑內(包含終點是已經回文))
        while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n && t[i - p[i] - 1] == t[i + p[i] + 1]) {
            p[i]++;
        }
        //Step 3. 新的回文是否有超過上一個r的限制邊界，如果有超過的話，更新新的r限制邊界與c中心
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
        //Step 4. 更新新的最大回文
        if (p[i] > maxLen) { //p[i] > maxLen：因為他的半徑已經包含"#"，所以剛好原本長度的2倍，所以最後可以直接用!
            maxLen = p[i];
            start = (i - maxLen) / 2; //記得要除以2，因為要把"#"去除掉 ("#a" 這樣是一組，所以除以2會是index)
        }
    }
    return s.substr(start, maxLen);//第二個參數超過s.size()沒差，他只是的擷取幾個元素(包含起始點)
}


string TwoPointer::ManachersAlg_iterator_5(string s) {
    int max_length = 1, start = 0;
    int rightmax = 0; int center = 0, now_radix = 0;
    //step 1.expand
    string s_expand = "#";
    for (auto it = s.begin(); it != s.end(); ++it)
        s_expand += *it, s_expand += "#";

    vector<int> p(s_expand.size(),0);

    auto longPalindrome = [&](auto it) {
        auto left = it - now_radix, right = it + now_radix;
        //在編譯器是不能只像begin()-1 ，連--left; 都不行(這就相當於linklist：head->next->next，但head->next = nullptr;)
        while (left > s_expand.begin() && right != s_expand.end() && *left == *right) {
            --left;
            ++right;
        }

        if (left == s_expand.begin() && *left == *right) {
            int len_temp = right - left;
            //step 3.Record the max_length & start
            if (max_length < len_temp) {
                start = left - s_expand.begin();
                max_length = len_temp;
            }
        }
        else {
            int len_temp = right - left - 1;
            //step 3.Record the max_length & start
            if (max_length < len_temp) {
                start = left - s_expand.begin() + 1;
                max_length = len_temp;
            }
        }
        int right_temp = right - s_expand.begin() - 1;
        if (right_temp > rightmax) {
            rightmax = right_temp;
            center = it - s_expand.begin();
        }
    };


    for (auto it = s_expand.begin(); it != s_expand.end(); ++it) {
        //step 4. Record is now_radix
        //int centertoit = (it - s_expand.begin()) - center; //int mirror = 2 * center - i; // 計算鏡像點(標準解法，他是用半徑)
        int i = it - s_expand.begin(); // 轉換 iterator 為索引
        //比較下面兩種，p[]他知道你在用指標運算變成int，但下面那個，他沒辦法保證it - s_expand.begin()是整數，除非你在前面加(int)(it - s_expand.begin())
        p[i] = i < rightmax ? min(p[(it - s_expand.begin()) - center], rightmax-i) : 0;
        //p[it - s_expand.begin()] = it - s_expand.begin() < rightmax ? min(p[centertoit], rightmax - (it - s_expand.begin())) : 0;

        //step 2.longestPalindrome
        longPalindrome(it);
    }

    //step 4. Reduce s_expand for s
    max_length = max_length >> 1;
    start = start >> 1;

    return s.substr(start, max_length);
}
#pragma endregion

#pragma region Leetcode 647. Palindromic Substrings
//Leetcode 647. Palindromic Substrings
int TwoPointer::Leetcode_Sol_647(string s, int _solution) {
    switch (_solution)
    {
    case 1:
        brute_force_647(s);
    case 2:
        return TwoPointer_647(s);
    case 3:
        return ManachersAlg_647(s);

    default:
        return 0; // 確保所有路徑都有回傳值
    }

    return 0;
}
/*暴力破解O(n^3)*/
int TwoPointer::brute_force_647(string s) {
    //n、n-1、n-2、n-3...1
    //再依次判斷是不是遞迴，是的話就count++
    int count = 0, strsize = s.size();
    for (int dist = 0; dist < strsize; dist++) {

        for (int i = 0; i < strsize; i++) {
            int l = i, r = l + dist; bool ispalindromic = true;
            while (l <= r) {
                if (r < strsize && s[l] == s[r])
                    r--, l++;
                else {
                    ispalindromic = false;
                    break;
                }
            }
            if (ispalindromic)  count++;
        }

    }

    return count;
}

/*中心擴展*/
int TwoPointer::TwoPointer_647(string s) {
    int count = 0;
    auto expand = [&](string& s, int left, int right) {
        int count = 0;
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            count++, left--, right++;
        }
        return count;
        };
    //這個i算是地塞
    for (int i = 0; i < s.size(); i++) {
        count += expand(s, i, i);
        count += expand(s, i, i + 1);
        //count += odd(s,i);
        //count += even(s,i);
    }
    return count;
}
int TwoPointer::odd(string s, int idx) {
    int l = idx, r = idx;
    int count = 0;
    while (l >= 0 && r < s.size() && s[l] == s[r]) {
        count++, l--, r++;
    }
    return count;
}
int TwoPointer::even(string s, int idx) {
    int l = idx, r = idx + 1;
    int count = 0;
    while (l >= 0 && r < s.size() && s[l] == s[r]) {
        count++, l--, r++;
    }
    return count;
}

/*Manachers*/
int TwoPointer::ManachersAlg_647(string s) {
    if (s.empty()) return 0;

    // Step 1: Transform the string into T with separators
    string T = "#";
    for (char c : s) {
        T += c;
        T += "#";
    }
    int n = T.size();

    // Step 2: Manacher's Algorithm
    vector<int> P(n, 0); // P[i] is the radius of palindrome centered at i
    int C = 0, R = 0; // Center and right boundary
    int count = 0;

    for (int i = 0; i < n; i++) {
        int mirror = 2 * C - i; // Mirror of i around C

        if (i < R)
            P[i] = min(R - i, P[mirror]);

        // Expand around i (別忘了是控制半徑!)
        while (i - P[i] - 1 >= 0 && i + P[i] + 1 < n && T[i - P[i] - 1] == T[i + P[i] + 1])
            P[i]++;

        // Update center and right boundary if expanded past R
        if (i + P[i] > R) {
            C = i;
            R = i + P[i];
        }

        // Step 3: Each palindrome centered at i contributes (P[i] + 1) / 2 real palindromes
        count += (P[i] + 1) / 2;//=> P[i]/2 + 1/2 (他(包含"#")就等於是只算他自己那層)
    }

    return count;
}

#pragma endregion
#pragma endregion























