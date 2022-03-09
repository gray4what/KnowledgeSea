
20.**Valid Parentheses**

=====

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
