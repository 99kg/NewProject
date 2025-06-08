from datetime import datetime  # 这样导入可以简化调用


class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False
        self.gender = "Unknown"
        self.age = "Unknown"
        self.attribute_history = []  # 记录属性变更历史

    def update_gender_age(self, gender, age):
        # 记录变更历史
        self.attribute_history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),  # 直接使用 datetime.now()
            "old_gender": self.gender,
            "new_gender": gender,
            "old_age": self.age,
            "new_age": age
        })
        self.gender = gender
        self.age = age