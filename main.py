from quantum import Quantum

if __name__ == '__main__':
    q = Quantum('cities/ref_tokyo.jpg', crop_faces=False)

    generated_images = q.generated_images(['cities/ref_tokyo.jpg', 'cities/fukuoka.jpg'])
    # generated_images = q.generated_images('faces')
