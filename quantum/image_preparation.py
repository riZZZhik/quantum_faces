import cv2
import dlib
import numpy as np
from PIL import Image
from loguru import logger
from resizeimage.resizeimage import resize_cover

TEMPLATE_INFO = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])


class ImagePreparation:
    def __init__(self, face_shape_predict_model=None):
        tpl_min, tpl_max = np.min(TEMPLATE_INFO, axis=0), np.max(TEMPLATE_INFO, axis=0)
        self.minmax_template = (TEMPLATE_INFO - tpl_min) / (tpl_max - tpl_min)

        if face_shape_predict_model:
            self.face_detector = dlib.get_frontal_face_detector()
            self.face_shape_predictor = dlib.shape_predictor(face_shape_predict_model)
        else:
            self.face_detector = None
            self.face_shape_predictor = None

        self.inner_eyes_and_bottom_lip = [39, 42, 57]

    def crop_faces(self, image):
        assert self.face_detector, "You didn't given face_shape_predict_model path to class"
        image_array = np.array(image)
        locations = self.get_face_detection(image_array)

        face_images = []
        if locations:
            for _, location in locations:
                rect = (location.left(), location.top(), location.right(), location.bottom())
                face_images.append(image.crop(rect))
        return face_images

    def get_face_detection(self, frame):
        """
        Detects faces on the image
        :param frame: input image
        :type frame: <type 'numpy.ndarray'>
        :return: enumerate list of rectangles of detected faces
        """
        assert self.face_detector, "You didn't given face_shape_predict_model path to class"

        detection = self.face_detector(frame, 1)
        if len(detection):
            return enumerate(detection)

    def get_face_shape_coordinates(self, face_image):
        """
        Gets coordinates of 68 points of face shape
        :param face_image: input image
        :type face_image: <type 'numpy.ndarray'>
        :return: list with 68-points  coordinates (x,y)
        """
        assert self.face_detector, "You didn't given face_shape_predict_model path to class"

        face_detection = self.get_face_detection(face_image)
        if face_detection is not None:
            for index, face_rectangle in face_detection:
                face_shape = self.face_shape_predictor(face_image, face_rectangle)
                return list(map(lambda part: (part.x, part.y), face_shape.parts()))

    def get_aligned_frame(self, face_image):
        """
        Gets aligned frame with centered shape of face
        :param face_image: input image
        :type face_image: <type 'numpy.ndarray'>
        :return: aligned frame
        """
        assert self.face_detector, "You didn't given face_shape_predict_model path to class"

        face_shape_coordinates = np.float32(self.get_face_shape_coordinates(face_image))
        if face_shape_coordinates is not None:
            face_shape_coordinates_indices = np.array(self.inner_eyes_and_bottom_lip)
            affine_transform = cv2.getAffineTransform(
                face_shape_coordinates[face_shape_coordinates_indices],
                face_image.shape[0] * self.minmax_template[face_shape_coordinates_indices])
            return cv2.warpAffine(face_image, affine_transform, face_image.shape[:2])

    def norm_images_from_disk(self, images_path, crop_type, resize_size=(128, 128)):
        assert type(resize_size) in (list, tuple) and len(resize_size) in (2, 3), \
            "resize_size type should be list or tuple and length == 2"
        assert crop_type or self.face_detector, "You didn't given face_shape_predict_model path to class"

        images = [Image.open(path) for path in images_path]
        norm_images = []

        if crop_type == 0:
            norm_images = [np.array(resize_cover(image, resize_size[:2])) for image in images]
        else:
            for image_path, image in zip(images_path, images):
                faces = self.crop_faces(image)

                if not faces:
                    logger.warning(f"No face found on {image_path} image")
                    norm_images.append(np.zeros(resize_size, np.int8))
                    continue

                if crop_type == 2:
                    aligned_face = self.get_aligned_frame(np.array(faces[0]))
                    if aligned_face:
                        shape = aligned_face.shape
                        coordinates = self.get_face_shape_coordinates(aligned_face)
                        if coordinates:
                            norm_images.append(np.zeros(shape[:2], np.int8))
                            size = int(shape[0] / 60)
                            for x, y in coordinates[-1]:
                                if shape[0] > y >= 0 and shape[1] > x >= 0:
                                    norm_images[-1] = cv2.circle(norm_images[-1], (x, y), size, 255, -1)

                            norm_images[-1] = Image.fromarray(norm_images[-1], "L")
                            norm_images[-1] = np.array(resize_cover(norm_images[-1], resize_size[:2]))
                        else:
                            logger.warning(f"No face found on {image_path} image")
                            norm_images.append(np.zeros(resize_size, np.int8))
                    else:
                        logger.warning(f"No face found on {image_path} image")
                        norm_images.append(np.zeros(resize_size, np.int8))
                else:
                    norm_images.append(np.array(resize_cover(faces[0], resize_size[:2])))

        return np.array(norm_images)
