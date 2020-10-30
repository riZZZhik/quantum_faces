from quantum import Quantum

if __name__ == '__main__':
    q = Quantum((32, 32))

    # generated_images = q.generate_images('faces')

    swap1 = q.swap_compare(['faces/0_0.jpg', 'faces/0_1.jpg'])
    print("Swap 1: ", swap1)
    swap2 = q.swap_compare(['faces/0_0.jpg', 'faces/1_0.jpg'])
    print("Swap 2: ", swap2)
