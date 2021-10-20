

# Evalutate
sexpr = 'sin(x) + x'
expr = parse_expr(sexpr)
x_vals = range(1, 40)
y_vals = [expr.subs('x', x_val).evalf() for x_val in x_vals]
known_dataset = list(zip(x_vals, y_vals))

heap = ExpressionHeap(heap=['Add', 'sin', 'x', 'x', None])
print(heap.evaluate(known_dataset))