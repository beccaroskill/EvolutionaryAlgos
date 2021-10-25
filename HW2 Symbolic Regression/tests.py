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


def swap_subtree(heap1, heap2, i1, i2):
    parents1 = [i1]
    parents2 = [i2]
    while parents2:
        parent1 = parents1.pop(0)
        print(parent1)
        parent2 = parents2.pop(0)
        heap1[parent1] = heap2[parent2]
        # add children for processing
        if 2*parent2 + 2 < len(heap2):
            parents2 += [2*parent2 + 1, 2*parent2 + 2]
        parents1 += [2*parent1 + 1, 2*parent1 + 2]
    return heap1
            

heap1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6]
heap2 = [9, 8, 7, 6, 5, 4, 3, 9, 8, 7, 6]
i1 = 2
i2 = 2

swapped = swap_subtree(heap1, heap2, i1, i2)

print(swapped)