class uo:
  def __init__(self, weight=0):
    self.weight = weight

  def cry(self):
    print(self.cryword())

class cat(uo):
  def __init__(self, weight, height):
    super().__init__(weight)
    self.height = height

  def cryword(self):
    return "meow"
  
  
u = cat(20, 60)
u.cry()
print(u.weight)