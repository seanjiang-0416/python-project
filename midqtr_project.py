"""
DSC 20 Mid-Quarter Project
Name: Zhengde Song/Zhixing Jiang
PID:  A16142930/A16400450

DSC 20 mid-Quarter Project that consists of three classes: RGBImage
ImageProcessing, and ImageKNNClassifier 
"""
def q3_doctests():
    """
    >>> test = RGBImage([[[1,2,3],[2,3,4],[3,4,5]],[[3,2,1],\
        [4,3,2],[5,4,2]],[[1,1,1],[2,2,2],[3,3,3]]])

    >>> print(type(test.copy().get_pixels()))    
"""
# Part 1: RGB Image #
class RGBImage:
    """
    a template for image objects in RGB color spaces.

    """

    def __init__(self, pixels):
        """
        A constructor that initializes a RGBImage instance and
        necessary instance variables.

        """
        self.pixels = pixels  # initialze the pixels list here

    def size(self):
        """
        A getter method that returns the size of the image, 
        where size is defined as a tuple of (number of rows, 
        number of columns).
        """
        return (len(self.pixels[0]),len(self.pixels[0][0]))

    def get_pixels(self):
        """
        A getter method that returns a DEEP COPY of the pixels
        matrix of the image (as a 3-dimensional list). This matrix
        of pixels is exactly the same pixels passed to the constructor.
        """
        output = []
        
        for x in range(len(self.pixels)):
            temp_1 = []
            for i in range(len(self.pixels[x])):
                temp_2 = []
                for t in self.pixels[x][i]:
                    temp_2.append(t)
                temp_1.append(temp_2)
            output.append(temp_1)
        
        return output
    def copy(self):
        """
        A method that returns a COPY of the RGBImage instance.
        """
        copy_instance=RGBImage(self.get_pixels())
        return copy_instance
        
    def get_pixel(self, row, col):
        """
        A getter method that returns the color of the pixel at 
        position (row, col).
        """
        if type(row)!=int or type(col)!=int:
            raise TypeError('not correct type')
        elif row>=len(self.pixels[0]) or row<0 or col>=len(self.pixels\
        [0][0]) or col<0:
            raise ValueError('not valid indices')
        else:
            return tuple([self.pixels[x][row][col] \
            for x in range (len(self.pixels))])
    
    def set_pixel(self, row, col, new_color):
        """
        A setter method that updates the color of the pixel at position
        (row, col) to the new_color in place.
        """
        if type(row)!=int or type(col)!=int:
            raise TypeError('not correct type')
        elif row>=len(self.pixels[0]) or row<0 or col>=len(self.pixels\
        [0][0]) or col<0:
            raise ValueError('not valid indices')
        for i in range(len(new_color)):
            if new_color[i]==-1:
                continue
            else:
                self.pixels[i][row][col]=new_color[i]

# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    contains several image processing methods
    """

    @staticmethod
    def negate(image):
        """
        A method that returns the negative image of the given image.

        """
        all_inversed=[[[(255-k) for k in j]for j in i] for i in\
             image.get_pixels()]
        return RGBImage(all_inversed)

    @staticmethod
    def tint(image, color):
        """
        A method that takes in a new color, and tints the given image
        using the new color.
        """
        
        tint_it=[[[(k+color[i])//2 for k in j] for j in\
         image.get_pixels()[i]] for i in\
             range(len(image.get_pixels()))]
        return RGBImage(tint_it)



    @staticmethod
    def clear_channel(image, channel):
        """
        A method that clears the given channel of the image.
        """

        image_matrix = image.get_pixels()
        new_matrix = [[[0 if x == channel else t for t in i]\
        for i in image_matrix[x]] for x in range (len(image_matrix))]
        return RGBImage(new_matrix)
        

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        A method that crops the image.
        """
        # YOUR CODE GOES HERE #
        a_copy_pixel=image.copy().get_pixels()
        a_copy_object=image.copy()
        if a_copy_object.size()[0]>=tl_row+target_size[0]\
            and a_copy_object.size()[1]>=tl_col+target_size[1]:
            
                
            after_crop=[[[a_copy_pixel[i][j][k] for k in \
                range(tl_col,target_size[1]+tl_col)] for j in \
                    range(tl_row,target_size[0]+tl_row)]\
                        for i in range(len(a_copy_pixel))]
            
            return RGBImage(after_crop)

        else: 
            if a_copy_object.size()[0]<tl_row+target_size[0]\
                and a_copy_object.size()[1]>=tl_col+target_size[1]:
                after_crop=[[[a_copy_pixel[i][j][k] for k in \
                range(tl_col,target_size[1]+tl_col)] for j in \
                    range(tl_row,a_copy_object.size()[0])]\
                        for i in range(len(a_copy_pixel))]

            elif a_copy_object.size()[1]<tl_col+target_size[1]\
                and a_copy_object.size()[0]>=tl_row+target_size[0]:
                after_crop=[[[a_copy_pixel[i][j][k] for k in \
                range(tl_col,a_copy_object.size()[1])] for j in \
                    range(tl_row,target_size[0]+tl_row)]\
                        for i in range(len(a_copy_pixel))]
            else:
                after_crop=[[[a_copy_pixel[i][j][k] for k in \
                range(tl_col,a_copy_object.size()[1])] for j in \
                    range(tl_row,a_copy_object.size()[0])]\
                        for i in range(len(a_copy_pixel))]
                        
            return RGBImage(after_crop)
    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        A method that performs the chroma key algorithm on 
        the chroma_image by replacing all pixels with the specified 
        color in the chroma_image to the pixels at the same places 
        in the background_image.
        """
        
        image_matrix = chroma_image.get_pixels()
        background_matrix = background_image.get_pixels()
        if isinstance(chroma_image, RGBImage)==False or \
            isinstance(background_image, RGBImage)==False:
                raise TypeError('not RGBImage instances')
        if (chroma_image.size() != background_image.size()):
            raise ValueError('Size not matches')
        
        for row in range(len(image_matrix[0])):
            for col in range(len(image_matrix[0][0])):
                if (tuple(chroma_image.get_pixel(row, col)) == color):
                    for i in range (len(image_matrix)):
                        image_matrix[i][row][col] = background_matrix[i][row][col]
            
        return RGBImage(image_matrix)
    # rotate_180 IS FOR EXTRA CREDIT (points undetermined)
    @staticmethod
    def rotate_180(image):
        """
        A method that rotates the image for 180 degrees.
        """
        
        image_matrix = image.get_pixels()
        
        new_matrix = [[[x[i][t] for t in range(len(x[i])-1, 0, -1)] \
        for i in range(len(x)-1, 0, -1)] for x in image_matrix]
        
        return RGBImage(new_matrix)
        

        

# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    A K-nearest Neighbors (KNN) classifier for the RGB images.
    """

    def __init__(self, n_neighbors):
        """
        A constructor that initializes a ImageKNNClassifier instance 
        and necessary instance variables.

        """
        self.n_neighhors=n_neighbors
        self.data=None
        
    def fit(self, data):
        """
        Fit the classifier by storing all training data in the 
        classifier instance.
        """
        if len(data)<=self.n_neighhors:
            raise ValueError('length not long enough')
        elif self.data!=None:
            raise ValueError('already have training data stored')
        self.data=data
    @staticmethod
    def distance(image1, image2):
        """
        A method to calculate the Euclidean distance between 
        RGB image image1 and image2.

        """
        if isinstance(image1, RGBImage)==False or \
            isinstance(image2, RGBImage)==False:
                raise TypeError('not RGBImage instances')
        if (image1.size() != image2.size()):
            raise ValueError('Size not matches')
        image1_pixels=image1.get_pixels()
        image2_pixels=image2.get_pixels()

        
        distances=[(image1_pixels[i][j][k]-image2_pixels[i][j][k])**2\

        for i in range(len(image1_pixels)) for j in \

        range(len(image1_pixels[i])) for k in range(len(image1_pixels[i][j]))]
        ans=sum(distances)**0.5
        
        return ans

        
    @staticmethod
    def vote(candidates):
        """
        Find the most popular label from a list of candidates 
        (nearest neighbors) labels.
        """
        # YOUR CODE GOES HERE #
        count=0
        popular_est=candidates[0]
        for i in candidates:
            frequency=candidates.count(i)
            if frequency>count:
                count=frequency
                popular_est=i
        return popular_est
        
    def predict(self, image):
        """
        Predict the label of the given image using the KNN 
        classification algorithm.
        """
        if self.data==None:
            raise ValueError('fit() method has NOT been called')
        nearest_neibours=sorted([(ImageKNNClassifier.distance(i[0],\
            image),i[1]) for i in self.data])[:self.n_neighhors]
        rank_labels=ImageKNNClassifier.vote([i[1] for i in nearest_neibours])    
        return rank_labels
        
