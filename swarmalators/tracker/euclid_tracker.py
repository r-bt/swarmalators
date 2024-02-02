import math

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def init(self, inital_positions):
        """
        Initalize the center points

        Args:
            inital_positions (list): The inital positions of the objects in form [x, y, w, h]
        """
        try:
            for position in inital_positions:
                x, y, w, h = position
                center_x = (x + x + w) // 2
                center_y = (y + y + h) // 2

                self.center_points[self.id_count] = (center_x, center_y)
                self.id_count += 1
        except Exception as e:
            print(inital_positions)
            raise RuntimeError("Inital positions must be in form [x, y, w, h]")

    def _get_closest_id(self, center_x, center_y):
        """
        Gets the closest id to the center point

        Args:
            center_x (int): The x coordinate of the center point
            center_y (int): The y coordinate of the center point

        Returns:
            int: The id of the closest point
        """
        closest_id = -1
        min_dist = math.inf

        for id, point in self.center_points.items():
            dist = math.sqrt((center_x - point[0]) ** 2 + (center_y - point[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_id = id

        return closest_id
    
    def update(self, objects_rect):
        """
        Update the center points of the objects

        Args:
            objects_rect (list): The bounding boxes of the objects in form [x, y, w, h]

        Returns:
            list: The new center points in form [id, (x, y)]
        """
        if len(objects_rect) != len(self.center_points):
            raise RuntimeError("Number of objects must match number of center points")

        new_center_points = {}

        ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            center_x = (x + x + w) // 2
            center_y = (y + y + h) // 2

            # Get the closet id to the center point
            closest_id = self._get_closest_id(center_x, center_y)

            ids.append(closest_id)

            # Assign the object id to the center point
            new_center_points[closest_id] = (center_x, center_y)

        if len(new_center_points) != len(self.center_points):
            print(len(objects_rect))
            print(new_center_points)
            print(self.center_points)
            raise RuntimeError("New center points length must match old center points")

        # Add the new center points to the existing center points
        self.center_points = new_center_points.copy()

        # Return the new center points sorted by id
        return sorted(new_center_points.items(), key=lambda x: x[0])