import math

class Point():

    def __init__(self , x : float = 0 , y : float = 0) -> None:
        self.x = x 
        self.y = y
    
    def set_x(self , new_x : float):
        self.x = new_x
    
    def set_y(self , new_y : float):
        self.y = new_y
    
    def calc_dist_to(self , p):
        return math.dist([p.x , p.y] , [self.x , self.y])

    def coord(self):
        print(f"<{self.x},{self.y}>")

    def quarter(self):
        if self.x > 0 and self.y > 0:
            print("first")
        if self.x < 0 and self.y > 0:
            print("second")
        if self.x < 0 and self.y < 0:
            print("third")
        if self.x > 0 and self.y < 0:
            print("forth")

# p1 = Point(1.4,4.4) # creating a new point
# p2 = Point(2.1,-5.2)

# print(p1.calc_dist_to(p2)) # calculating the distance
# p1.coord() # printing the coordinate
# p2.quarter() # printing the quarter

class DataStruct():

    def __init__(self , dataset : list) -> None:
        self.dataset = dataset
        self.y = dataset[-1]
        dataset.pop(-1)
        self.x = dataset
    
# Dataset = [[4, 3, 5, 4, 0],
#             [1, 3, 2, 7, 0],
#             [8, 1, 5, 8, 0],
#             [9, 3, 5, 8, 1],
#             [0, 7, 2, 7, 0],
#             [0, 3, 5, 4, 1]]

# ds = DataStruct(dataset = Dataset)
# print(ds.y)
# print(ds.x)

