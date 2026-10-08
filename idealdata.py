# GENERATED FILE, made by tools/make_ideal_face.py. Do not edit by hand: run the tool again (see the README).
# The ideal face the app rates against (see idealface.py), worked out from the front photos of the Face Research
# Lab London Set (DeBruine & Jones 2017, CC BY 4.0, https://doi.org/10.6084/m9.figshare.5047666) and the model
# face pictures.

# The idealface.ANCHORS of the average face (x, y, depth), outer eye corners 1 apart
REFERENCE = [
    [0, 0.13348, -0.09707],
    [0, 0.045873, -0.049273],
    [0, 0.21245, -0.1533],
    [0, 0.28488, -0.21528],
    [-0.18699, 0.10168, 0.14537],
    [0.18699, 0.10168, 0.14537],
    [-0.5, 0.078626, 0.20654],
    [0.5, 0.078626, 0.20654],
    [0, -0.51002, -0.0074717],
    [0, -0.33503, -0.047953],
    [0, -0.14538, -0.075781],
    [0, -0.046867, -0.057688],
]

# The expression (idealface.EXPRESSIONS: smile, squint, mouthOpen, upperLipUp) of a typical neutral face
NEUTRAL_EXPRESSION = [0.044514, 0.26957, 0.0043033, 0.0038199]

# How much each measurement changes per unit (0 to 1) of each expression, in the order of NEUTRAL_EXPRESSION
EXPRESSION_SLOPES = {
    "eyeSpacing": [-0.00017012, 0.0030793, 0.00035126, 0.002525],
    "innerEyeGap": [0.039787, 0.079257, 0.004952, 0.01644],
    "canthalTilt": [-0.0042294, 0.0014787, 0.0073478, 0.0066419],
    "eyeSize": [-0.00299, -0.008103, -0.0011581, -0.00074935],
    "browHeight": [0.0031904, -0.010074, -0.0026192, 0.0001563],
    "browArch": [0.001499, -0.0027616, -0.0017083, -0.0046246],
    "browTilt": [-0.011137, -0.011359, 0.0075932, 0.014059],
    "noseWidth": [0.10926, 0.052991, 0.0082191, 0.041057],
    "noseLength": [-0.0079023, -0.0041498, -0.0074914, -0.00062306],
    "lipFullness": [-0.03111, -0.0050907, -0.029953, -0.038045],
    "upperLip": [-0.023631, -0.025647, -0.082539, 0.055332],
    "jawSharpness": [-0.0010099, -0.0014258, 0.018035, 0.02235],
    "faceLength": [0.01871, -0.0022699, 0.020253, 0.0095954],
    "jawWidth": [0.0066061, 0.0018765, -0.0021877, 0.00026241],
    "chinWidth": [-0.0036888, 0.0029219, 0.0012986, 0.0036143],
    "chinHeight": [0.011497, -0.0080329, -0.048083, -0.066765],
    "thirds": [-0.042624, -0.035324, -0.044793, -0.033668],
}

# The model face pictures the targets were moved towards
MODEL_PICTURES = {'boy': ['perBoy.jpg', 'perBoy2.jpg', 'perBoy3.jpg', 'perBoy4.jpg', 'perBoy5.jpg'], 'girl': ['perGirl.jpg', 'perGirl2.jpg', 'perGirl3.jpg', 'perGirl4.jpg', 'perGirl5.jpg']}

# For each gender and measurement: (target, tolerance). The target is the average of real faces moved towards the
# model faces, the tolerance is how much real faces differ (one standard deviation)
IDEALS = {
    "boy": {
        "eyeSpacing": (0.459, 0.024477),
        "innerEyeGap": (1.1743, 0.084041),
        "canthalTilt": (0.060659, 0.027277),
        "eyeSize": (0.20198, 0.0098623),
        "browHeight": (0.19892, 0.025116),
        "browArch": (0.072877, 0.0060405),
        "browTilt": (-0.06083, 0.016065),
        "noseWidth": (1.2364, 0.066776),
        "noseLength": (0.2964, 0.010724),
        "lipFullness": (0.25986, 0.060838),
        "upperLip": (0.48454, 0.167),
        "jawSharpness": (0.2, 0.019461),
        "faceLength": (1.2296, 0.044667),
        "jawWidth": (0.89082, 0.017424),
        "chinWidth": (0.1895, 0.0078718),
        "chinHeight": (0.52136, 0.05283),
        "thirds": (0.96733, 0.055191),
    },
    "girl": {
        "eyeSpacing": (0.47297, 0.020524),
        "innerEyeGap": (1.1969, 0.084896),
        "canthalTilt": (0.1125, 0.021847),
        "eyeSize": (0.2081, 0.0094495),
        "browHeight": (0.2262, 0.027897),
        "browArch": (0.082199, 0.0065287),
        "browTilt": (-0.046473, 0.014346),
        "noseWidth": (1.1179, 0.067794),
        "noseLength": (0.30625, 0.0074793),
        "lipFullness": (0.32593, 0.045899),
        "upperLip": (0.65855, 0.15125),
        "jawSharpness": (0.20753, 0.015793),
        "faceLength": (1.2007, 0.045343),
        "jawWidth": (0.8594, 0.015777),
        "chinWidth": (0.17136, 0.0059597),
        "chinHeight": (0.46434, 0.039352),
        "thirds": (1.0905, 0.055359),
    },
}
