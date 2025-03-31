
```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        st = []
        result = [0] * len(temperatures)
        st.append(0)
        for i in range(1, len(temperatures)):
            if temperatures[i] <= temperatures[st[-1]]:
                st.append(i)
            else:
                while st and temperatures[i] > temperatures[st[-1]]:
                    result[st[-1]] = i - st[-1]
                    st.pop()
                st.append(i)
        return result
```
