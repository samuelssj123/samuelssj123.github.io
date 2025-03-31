List: 42. 接雨水

[42. 接雨水trapping-rain-water](#01)，[](#02)

# <span id="01">42. 接雨水trapping-rain-water</span>

[Leetcode](https://leetcode.cn/problems/trapping-rain-water/description/) 

[Learning Materials](https://programmercarl.com/0042.%E6%8E%A5%E9%9B%A8%E6%B0%B4.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/42-trapping-rain-water.png)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        st = []
        result = 0
        st.append(0)
        for i in range(1, len(height)):
            if height[i] <= height[st[-1]]:
                st.append(i)
            else:
                while st and height[i] > height[st[-1]]:
                    mid = st[-1]
                    st.pop()
                    if st:
                        h = min(height[i], height[st[-1]]) - height[mid]
                        w = i - st[-1] - 1
                        result += h * w
                st.append(i)
        return result
```

# <span id="02">84.柱状图中最大的矩形largest-rectangle-in-histogram</span>

[Leetcode](https://leetcode.cn/problems/largest-rectangle-in-histogram/description/) 

[Learning Materials](https://programmercarl.com/0084.%E6%9F%B1%E7%8A%B6%E5%9B%BE%E4%B8%AD%E6%9C%80%E5%A4%A7%E7%9A%84%E7%9F%A9%E5%BD%A2.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/84-largest-rectangle-in-histogram.png)

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        st = []
        result = 0
        st.append(0)
        heights.append(0)
        heights.insert(0, 0)
        for i in range(1, len(heights)):
            if heights[i] >= heights[st[-1]]:
                st.append(i)
            else:
                while st and heights[i] < heights[st[-1]]:
                    mid = st[-1]
                    st.pop()
                    if st:
                        h = heights[mid]
                        w = i - st[-1] - 1
                        result = max(result, h * w)
                st.append(i)
        return result       
  ```
