from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr

x = symbols('x')
      
from collections import deque 
import math
import numpy as np 

class ExpressionHeap:
    
    valid_operators = ['Add', 'Mul', 'Sub', 'Div', 'sin', 'cos']
    all_operators = ['Add', 'Mul', 'Pow', 'Sub', 'Div', 'sin', 'cos']
    unary_operators = ['sin', 'cos']
    
    def __init__(self, expr=None, heap=[]):
        if expr:
            self.heap = ExpressionHeap.from_expr(expr)
        else:
            self.heap = heap
    
    def separate_str_args(args):
        comma_i = [i for i, x in enumerate(args) if x == ',']
        for i in comma_i:
            open_count = len([c for c in args[:i] if c == '('])
            closed_count = len([c for c in args[:i] if c == ')'])
            if open_count == closed_count:
                left = args[:i]
                right = args[i+1:].strip()
                return (left, right)
            
    def is_legal_operation(op, left, right):
        legal = True
        if op == 'Pow' and (left == 'x' or float(left) > 2) and right == 'x':
            legal = False
        elif op == 'root' and right == 'x':
            legal = False
        return legal
    
    def build_heap(expr, heap, parent_i):
        if ',' in expr:
            operator = expr[:expr.find('(')]
            heap[parent_i] = operator
            left, right = ExpressionHeap.separate_str_args(expr[expr.find('(')+1: expr.rfind(')')])
            heap = ExpressionHeap.build_heap(left, heap, 2*parent_i + 1)
            heap = ExpressionHeap.build_heap(right, heap, 2*parent_i + 2)
        else:
            # We might need to allocate more memory if tree is unbalanced
            if len(heap) <= parent_i:
                heap += [None] * (parent_i - len(heap) + 1)
            heap[parent_i] = expr
        return heap
    
    def from_expr(expr):
        num_ops = len([i for i, x in enumerate(expr) if x == ','])
        
        # Best case, our tree is balanced (we may need to allocate more later)
        heap = [None] * (2 * num_ops + 1)
        heap = ExpressionHeap.build_heap(expr, heap, 0)
        return heap
    
    def to_expr(self, include_sub_expr=False):
        arg_stack = deque() 
        str_expr = ''
        heap = self.heap.copy()
        sub_expr = {} if include_sub_expr else None
        
        # Deal with expressions that are just one term, no operators
        if len(heap) == 1:
            str_expr = str(heap[0])
            if include_sub_expr:
                sub_expr[0] = str_expr
                
        for i in range(len(heap)-1, 0, -1):
            x = heap[i]
            if x not in ExpressionHeap.all_operators:
                if arg_stack:
                    parent_i = math.floor((i - 1)/2)
                    operator = heap[parent_i]
                    left, right = [x, arg_stack.pop()]
                    if operator is not None and left is not None:
                        if operator == 'Sub':
                            str_expr = 'Add({}, Mul(-1, {}))'.format(left, right)
                        elif operator == 'Div':
                            str_expr = 'Mul({}, Pow({}, -1))'.format(left, right)
                        elif operator in ExpressionHeap.unary_operators:
                            if left and right:
                                print('left', left, 'right', right)
                            str_expr = '{}({})'.format(operator, left)
                        else:
                            str_expr = '{}({}, {})'.format(operator, left, right)
                        if include_sub_expr:
                            sub_expr[parent_i] = str_expr
                        if i > 0:
                            heap[parent_i] = str_expr
                else:
                    arg_stack.append(x)
                    
        return (str_expr, sub_expr) if include_sub_expr else str_expr
        
    def evaluate(self, data):
        str_expr = self.to_expr()
        try:
            expr = parse_expr(str_expr)
            sse = 0
            for x_val, y_val in data:
                f_val = expr.subs('x', x_val).evalf()
                e_sq = (f_val - y_val)**2
                sse += e_sq
        # In case expression can be reduced to nothing
        except ValueError:
            sse = sum([y_val**2 for _, y_val in data])
        imaginary = [t for t in ['zoo', 'oo', '-oo', 'I', 'nan'] if t in str(sse)]
        if imaginary:
            return (False, None)
        else:
            mse = sse/len(data)
            return (True, mse)
    
    def replace_subtree(self, subroot, subtree):
        # TODO - Expand heap if subtree requires it
        
        # TODO - Replace subtree
        # for i in (len(subtree) - 1 / 2):
        # For now, just assume replacing subtree with a constant
        self.heap[subroot] = subtree[0]
    
        # Remove extraneous children
        for i in range(len(self.heap)):
            parent_i = math.floor((i - 1)/2)
            if parent_i > 0 and self.heap[parent_i] not in ExpressionHeap.all_operators:
                self.heap[i] = None 
            elif parent_i == 0 and len(self.heap) == 3:
                self.heap = [ self.heap[0] ]

        expr = self.to_expr()
        self.heap = ExpressionHeap.from_expr(expr)
        
    def trim_heap(self, data, threshold=0.1):

        _, sub_expr = self.to_expr(include_sub_expr=True)
        for subroot in sub_expr:
            str_expr = sub_expr[subroot]
            expr = parse_expr(str_expr)
            f_vals = [expr.subs('x', x_val).evalf() for x_val, _ in data]
            var = np.var(f_vals)
            if var < threshold:
                mean = np.mean(f_vals)
                self.replace_subtree(subroot, [mean])