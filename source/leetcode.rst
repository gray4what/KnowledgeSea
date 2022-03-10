##################
Leetcode
##################

20. **Valid Parentheses**
==========================

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
An input string is valid if:
Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.

.. code-block:: python

  class Solution:
      def isValid(self, s: str) -> bool:
          char_map = {")": "(",
                      "}":"{",
                      "]":"["}

          visit = collections.deque()

          if len(s) <= 1: return False

          visit.append(s[0])
          for char in s[1:]:
              if char in char_map:
                  if visit and visit[-1] == char_map[char]:
                      visit.pop()

                  else:
                      return False

              else:
                  visit.append(char)

          return True if not visit else False


.. code-block:: python

    class Solution:
      def isValid(self, s: str) -> bool:
          left = []
          leftOf = {
              ')':'(',
              ']':'[',
              '}':'{'
          }
          for c in s:
              if c in '([{':
                  left.append(c)
              elif left and leftOf[c]==left[-1]: # 右括号 + left不为空 + 和最近左括号能匹配
                  left.pop()
              else: # 右括号 + （left为空 / 和堆顶括号不匹配）
                  return False

          # left中所有左括号都被匹配则return True 反之False
          return not left



146. **LRU cache**
==================

.. code-block:: python

    from collections import OrderedDict
    class LRUCache(OrderedDict):

        def __init__(self, capacity):
            """
            :type capacity: int
            """
            self.capacity = capacity

        def get(self, key):
            """
            :type key: int
            :rtype: int
            """
            if key not in self:
                return - 1
            
            self.move_to_end(key)
            return self[key]

        def put(self, key, value):
            """
            :type key: int
            :type value: int
            :rtype: void
            """
            if key in self:
                self.move_to_end(key)
            self[key] = value
            if len(self) > self.capacity:
                self.popitem(last = False)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class DLinkedNode(): 
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None
            
     class LRUCache():
         def _add_node(self, node):
             """
             Always add the new node right after head.
             """
             node.prev = self.head
             node.next = self.head.next

             self.head.next.prev = node
             self.head.next = node

         def _remove_node(self, node):
             """
             Remove an existing node from the linked list.
             """
             prev = node.prev
             new = node.next

             prev.next = new
             new.prev = prev

         def _move_to_head(self, node):
             """
             Move certain node in between to the head.
             """
             self._remove_node(node)
             self._add_node(node)

         def _pop_tail(self):
             """
             Pop the current tail.
             """
             res = self.tail.prev
             self._remove_node(res)
             return res

         def __init__(self, capacity):
             """
             :type capacity: int
             """
             self.cache = {}
             self.size = 0
             self.capacity = capacity
             self.head, self.tail = DLinkedNode(), DLinkedNode()

             self.head.next = self.tail
             self.tail.prev = self.head
             

         def get(self, key):
             """
             :type key: int
             :rtype: int
             """
             node = self.cache.get(key, None)
             if not node:
                 return -1

             # move the accessed node to the head;
             self._move_to_head(node)

             return node.value

         def put(self, key, value):
             """
             :type key: int
             :type value: int
             :rtype: void
             """
             node = self.cache.get(key)

             if not node: 
                 newNode = DLinkedNode()
                 newNode.key = key
                 newNode.value = value

                 self.cache[key] = newNode
                 self._add_node(newNode)

                 self.size += 1

                 if self.size > self.capacity:
                     # pop the tail
                     tail = self._pop_tail()
                     del self.cache[tail.key]
                     self.size -= 1
             else:
                 # update the value.
                 node.value = value
                 self._move_to_head(node)


875. **Koko Eating Bananas**
==============================================

.. code-block:: python 

      import math
      class Solution:
          def minEatingSpeed(self, piles, H):
              # 初始化起点和终点， 最快的速度可以一次拿完最大的一堆
              start = 1
              end = max(piles)
              
              # while loop进行二分查找
              while start + 1 < end:
                  mid = start + (end - start) // 2
                  
                  # 如果中点所需时间大于H, 我们需要加速， 将起点设为中点
                  if self.timeH(piles, mid) > H:
                      start = mid
                  # 如果中点所需时间小于H, 我们需要减速， 将终点设为中点
                  else:
                      end = mid
                      
              # 提交前确认起点是否满足条件，我们要尽量慢拿
              if self.timeH(piles, start) <= H:
                  return start
              
              # 若起点不符合， 则中点是答案
              return end
                  
              
              
          def timeH(self, piles, K):
              # 初始化时间
              H = 0
              
              #求拿每一堆需要多长时间
              for pile in piles:
                  H += math.ceil(pile / K)
                  
              return H


1011. **Capacity To Ship Packages Within D Days**
==================================================

.. code-block:: python

    def shipWithinDays(weights: List[int], D: int) -> int:
    def feasible(capacity) -> bool:
        days = 1
        total = 0
        for weight in weights:
            total += weight
            if total > capacity:  # too heavy, wait for the next day
                total = weight
                days += 1
                if days > D:  # cannot ship within D days
                    return False
        return True

    left, right = max(weights), sum(weights)
    while left < right:
        mid = left + (right - left) // 2
        if feasible(mid):
            right = mid
        else:
            left = mid + 1
    return left

392. **Is Subsequence**
=========================

.. code-block:: python 

     class Solution:
         def isSubsequence(self, s: str, t: str) -> bool:
             LEFT_BOUND, RIGHT_BOUND = len(s), len(t)

             p_left = p_right = 0
             while p_left < LEFT_BOUND and p_right < RIGHT_BOUND:
                 # move both pointers or just the right pointer
                 if s[p_left] == t[p_right]:
                     p_left += 1
                 p_right += 1

             return p_left == LEFT_BOUND

234. **Palindrome Linked List**
================================

.. code-block:: python 

     def isPalindrome(self, head):
         fast = slow = head
         # find the mid node
         while fast and fast.next:
             fast = fast.next.next
             slow = slow.next
         # reverse the second half
         node = None
         while slow:
             nxt = slow.next
             slow.next = node
             node = slow
             slow = nxt
         # compare the first and second half nodes
         while node: # while node and head:
             if node.val != head.val:
                 return False
             node = node.next
             head = head.next
         return True

Palindrome string 
----------------------

.. code-block:: c++

    bool isPalindrome(string s) {
        int left = 0, right = s.length - 1;
        while (left < right) {
            if (s[left] != s[right])
                return false;
            left++; right--;
        }
        return true;
    }


26. **remove duplication in array**
====================================

.. code-block:: python 

    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        
        if n == 0:
            return 0
        
        slow, fast = 0, 1
        
        while fast < n:
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
                
            fast += 1
            
        return slow + 1

83. **remove duplication in linked-list**
==========================================

.. code-block:: python 

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        slow, fast = head, head.next
        
        while fast:
            if fast.val != slow.val:
                slow.next = fast
                slow = slow.next
                
            fast = fast.next

        # 断开与后面重复元素的连接   
        slow.next = None
        return head

77. **Combination** 
===============================

.. code-block:: python

    class Solution:
        def combine(self, n: int, k: int) -> List[List[int]]:
            def backtrack(first = 1, curr = []):
                # if the combination is done
                if len(curr) == k:  
                    output.append(curr[:])
                for i in range(first, n + 1):
                    # add i into the current combination
                    curr.append(i)
                    # use next integers to complete the combination
                    backtrack(i + 1, curr)
                    # backtrack
                    curr.pop()
            
            output = []
            backtrack()
            return output

46. **Permutation** 
=====================

.. code-block:: python 

    class Solution:
        def permute(self, nums: List[int]) -> List[List[int]]:
            
            res =[]
            permu = []
            
            counter = {n:0 for n in nums}
            for n in nums:
                counter[n] +=1
                
            
            def dfs():
                
                if len(permu) == len(nums):
                    return res.append(permu.copy())
                    
                for n in counter:
                    if counter[n] > 0:
                        permu.append(n)
                        counter[n] -= 1

                        dfs()

                        counter[n] += 1

                        permu.pop()
                            
            dfs()
            return res

    
47. **Permutation II**
======================
.. code-block:: python 

    class Solution:
        def permuteUnique(self, nums: List[int]) -> List[List[int]]:
            res = []
            permu = []
            count = {x:0 for x in nums}
            for i in nums:
                count[i] +=1
            
            def helper(count):
                if len(permu) == len(nums):
                    return res.append(permu[:])
                
                
                for n in count:
                    if count[n] > 0:
                        permu.append(n)
                        
                        count[n] -=1 
                        
                        helper(count)
                        
                        count[n] += 1
                        permu.pop()
            helper(count)            
            return res

48. **Minimum Window Substring**
=================================

.. code-block:: python 

    class Solution:
        def minWindow(self, s: str, t: str) -> str:
            # A D O B E C O D E B A N C
            
            window = {}
            
            need = {}
            
            if not t or not s:
                return ""
            
            for st in t:  # creata a counter for t.  S: O(n)  Speed: O(1)   
                need[st] = need.get(st, 0) + 1 
            
            left = right = 0
            valid = 0
            len_window = 100000
            
            ans = (100000, None, None) # window len, left, right
            while right < len(s):
                char = s[right]
                # if need.get(char): # add wanted char to window
                window[char] = window.get(char, 0) + 1
                    
                if char in need and window[char] == need[char]:
                    valid += 1 
                        
                # if need to shrink left 
                while left <= right and valid == len(need):
                    # if right - left < len_window:
                    #     start = left
                    #     len_window = right - left      
                    char2 = s[left]
                    window[char2] -=1  # remove most left 

                    if right - left + 1 < ans[0]:   # update min window
                        ans = (right - left + 1, left, right)
                    if char2 in need and window[char2] < need[char2]:
                        valid -= 1
                    left +=1   
                right += 1  # right shift window

            return "" if ans[0] == 100000 else s[ans[1]:ans[2] + 1]

204. **Count Prime** 
====================

.. code-block:: python

    def countPrimes(self, n):
        if n <= 2:
            return 0
        dp = [True] * n
        dp[0] = dp[1] = False
        for i in range(2, n):
            if dp[i]:
                for j in range(i*i, n, i):
                    dp[j] = False
        return sum(dp)

42. **Trapping water**
========================

.. code-block:: python

    def trap(self, height: List[int]) -> int:
    
        res = 0
        
        left = 0 
        n = len(height)
        right = n - 1
        
        l_max = r_max = 0
        while left < right:
            l_max = max(l_max, height[left])
            r_max = max(r_max, height[right])
            
            if l_max < r_max:
                res += l_max - height[left]
                left +=1
            else:
                res += r_max - height[right]
                right -=1
            
        return res

11. **Contain With Most Water**
==================================

.. code-block:: python 

    def maxArea(self, height: List[int]) -> int:
            
        pl = 0
        pr = len(height) - 1
        res = 0
        
        while pl <= pr:
            area = (pr - pl) * min(height[pr], height[pl]) 
            res = max(area, res)
            
            if height[pl] < height[pr]:
                pl +=1
            else:
                pr -=1
    
            
        return res


417. **Pacific Atlantic Water Flow**
=====================================

.. code-block:: python 

    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        if not matrix:
            return []
        p_land = set()
        a_land = set()
        R, C = len(matrix), len(matrix[0])
        def spread(i, j, land):
            land.add((i, j))
            for x, y in ((i+1, j), (i, j+1), (i-1, j), (i, j-1)):
                if (0<=x<R and 0<=y<C and matrix[x][y] >= matrix[i][j]
                        and (x, y) not in land):
                    spread(x, y, land)

        for i in range(R):
            spread(i, 0, p_land)
            spread(i, C-1, a_land)
        for j in range(C):
            spread(0, j, p_land)
            spread(R-1, j, a_land)
        return list(p_land & a_land)


84. **Largest Rectangle in Histogram**
=======================================

.. code-block:: python

    def largestRectangleArea(self, heights: List[int]) -> int:
        maxArea = 0

        stack = []

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                idx, height = stack.pop()
                maxArea = max(maxArea, height * (i - idx))
                start = idx
            stack.append((start, h))                        # 计算左到右, 高度递减
            
        for i, h in stack:
            maxArea = max(maxArea, h * (len(heights) - i ))  # 计算最低高度, 从右到左
            
        return maxArea

5. **Longest Palindrome Substring**
=======================================

.. code-block:: python

    def longestPalindrome(self, s: str) -> str:
        max_len = 0
        res = ""
        for i in range(len(s)):
            l = r = i               # odd            
            while l >=0 and r < len(s) and s[l] == s[r]:
                if (r -l + 1) > max_len:
                    res = s[l:r+1]
                    max_len = r - l + 1
                l -= 1
                r += 1

            l, r = i, i + 1
            while l >=0 and r < len(s) and s[l] == s[r]:
                if (r -l + 1) > max_len:
                    res = s[l:r+1]
                    max_len = r - l + 1
                l -= 1
                r += 1
        return res           

398. **Random Pick Index**
==============================

.. code-block:: python 

    class Solution:
        
        def __init__(self, nums):
            self.nums = nums
            

        def pick(self, target):
        return random.choice([k for k, v in enumerate(self.nums) if v == target])

382. Random pick in linked-list
----------------------------------

.. code-block:: python 

    class Solution:
        def __init__(self, head):
            self.head = head

        def getRandom(self):
            n, k = 1, 1
            head, ans = self.head, self.head
            while head.next:
                n += 1
                head = head.next
                if random.random() < k/n:
                    ans = ans.next
                    k += 1
                    
            return ans.val

448. **Missing number**
=======================

.. code-block:: python 

    def missingNumber(self, nums: List[int]) -> int:
        #思路3，防止整形溢出的优化
        res = len(nums)
        for i,num in enumerate(nums):
            res+=i-num
            return res


.. code-block:: python 

    def missingNumber(self, nums: List[int]) -> int:
        #思路2，求和
        n = len(nums)
        return n*(n+1)//2-sum(nums)


645. **Set Mismatch**
========================

.. code-block:: python 

    def findErrorNums(self, nums: List[int]) -> List[int]:
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        s = n*(n+1)//2
        miss = s - sum(set(nums))
        duplicate = sum(nums) + miss - s
        return [duplicate, miss]



