from symbolic_regression import ExpressionHeap, SearchAlgorithms, load_dataset
from sympy.parsing.sympy_parser import parse_expr      

# Check function has 0 error with itself
def err_identity():
    sexpr = 'sin(x) + x'
    expr = parse_expr(sexpr)
    x_vals = range(1, 40)
    y_vals = [expr.subs('x', x_val).evalf() for x_val in x_vals]
    known_dataset = list(zip(x_vals, y_vals))
    
    heap = ExpressionHeap(heap=['Add', 'sin', 'x', 'x', None])
    print(heap.evaluate(known_dataset))
    
def make_random_heap():
    random_search = SearchAlgorithms()
    random_heap = random_search.get_random_heap()
    print(random_heap.heap)
    
def evaluate_random_heap():
    dataset = load_dataset('data.txt')
    random_search = SearchAlgorithms()
    random_heap = random_search.get_random_heap()
    print(random_heap.evaluate(dataset))

evaluate_random_heap()

