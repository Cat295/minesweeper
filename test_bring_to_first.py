def bring_to_front(a, b, c):
    # For list a
    for element in c:
        if element in a:
            a.remove(element)
            a.insert(0, element)
    
    # For list b
    for element in c:
        if element in b:
            b.remove(element)
            b.insert(0, element)
    
    return a, b

# Example usage
a = [1, 2, 3, 4, 5]
b = ['x', 'y', 'z']
c = [3, 'y']

a, b = bring_to_front(a, b, c)
print("Updated list a:", a)
print("Updated list b:", b)
