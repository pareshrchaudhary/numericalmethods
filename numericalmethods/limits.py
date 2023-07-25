import math

class Limits():
  def __init__(self, epsilon=1e-6):
    self.epsilon = epsilon

  def calculate_limit(self, func, point):
    return func(point - 1e-6), func(point + 1e-6)

  def evaluate_limit(self, func, point):
    left_limit, right_limit = self.calculate_limit(func, point)

    if (math.isclose(left_limit, right_limit, rel_tol =1e-4)):
      return True, round(((left_limit + right_limit) / 2), 4)
    else:
      return False, None

  def add(self, func1, func2, point):
    def add_function(x):
      return func1(x) + func2(x)
    return self.evaluate_limit(add_function, point)

  def sub(self, func1, func2, point):
    def sub_function(x):
      return func1(x) - func2(x)
    return self.evaluate_limit(sub_function, point)

  def mul(self, func1, func2, point):
    def mul_function(x):
      return func1(x) * func2(x)
    return self.evaluate_limit(mul_function, point)

  def div(self, func1, func2, point):
    def div_function(x):
      if func2(x) == 0:
        raise ValueError("Division by zero is not allowed.")
      return func1(x) / func2(x)
    return self.evaluate_limit(div_function, point)

if (__name__ == '__main__'):
  def f(x):
    return x**2
  
  lim = Limits()
  limit_f = lim.evaluate_limit(f, 5)[1]
