import numpy as np

class LogicGate:
    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.th = None
        self.out = None
        self.x1 = self.x2 = None


    def print_output(self, gate):
        if gate == 'AND':
            print(f'{self.x1} AND {self.x2} is {self.out}')
        elif gate == 'OR':    
            print(f'{self.x1} OR {self.x2} is {self.out}')
        elif gate == 'NAND':    
            print(f'{self.x1} NAND {self.x2} is {self.out}')   
        elif gate == 'NOR':    
            print(f'{self.x1} NOR {self.x2} is {self.out}')   

    def or_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.0

        self.x1 = x1
        self.x2 = x2

        # use numpy to perform matrix multiplication
        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        if np.sum(x*w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0        



    def and_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.99

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        
        if np.sum(x*w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0
        
    def nand_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.99

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        
        if np.sum(x*w) < self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0


    def nor_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.1

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        if np.sum(x*w) < self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0     




# Code to display message about the class and how to use the class
if __name__ == '__main__':
    print(''' This class contains basic logic gates used in Electrical Engineering, which are:
          1) OR gate
          2) NOR gate
          3) AND gate
          4) NAND gate 
          
          To use this class, user can use the following syntax:

          import logic_gate as lg
          logic_gate = lg.logic_gate()
          logic_gate.(GATE NAME)(x,y)
          Where GATE NAME is the type of gate you want to use. It should be replaced with one of the following
          - and_gate 
          - nand_gate
          - or_gate
          - nor_gate

          Whereas, x and y are the binary inputs (1 or 0), that you want to perform logic operations on.

          Here is an example:
          logic_gate.or_gate(1,0)

          This line of code will return a value of 1 

          Additionally, the class has built-in functioin that allows last output.
          This can be done easily by following syntax 

          logic_gate.print_output('NOR')

          NOR gate is used as an example and may be replaced with the user desired 
          targer logic gate's name


          ''')
    
