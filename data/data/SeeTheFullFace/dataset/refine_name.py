import os
import shutil
import glob
import unidecode

class spellingPolice():
    """
    The police with his ability to check your pronunciation 
    """
    def __init__(self):
        self.decoder = unidecode.unidecode

    def check_special_characters(self, path):
        """
        Check if the given path contains special characters. If it does, rename it
        Args:
            path: path to check: str
        Return: 
            renamed_path
        """
        return path.replace("&", "and").replace(" ", "_").replace("'", ".").replace("-", "_")

    def check_decoded(self, path):
        """
        Check if the given path is decoded. If it's not, decode it and return
        Args:
            path: path to check: str
        Return:
            renamed_path: str
        """
        return self.decoder(path)

    def check_duplicated(self, path):
        """
        Given a path with subfolders as classes. Check if there are any duplicated classes. Delete if there are
        In the senario, duplicated are named with "-2" at the end, delete it and rename the "-1" to its origin
        Args:
            path: path with subfolder: str
        Return:
            A report
        """
        classes = []
        count_1 = 0
        count_n = 0
        for c in os.listdir(path):
            if ".txt" not in classes:
                classes.append(c)
        for c in classes:
            if "-1" in c:
                os.rename(os.path.join(path, c), os.path.join(path, c.replace("-1", "")))
                count_1 += 1
            if "-2" in c or "-3" in c or "-4" in c :
                shutil.rmtree(os.path.join(path, c))
                count_n += 1
        print("{} subfolders have been renamed, {} others have been deleted".format(count_1, count_n))

    def check_match_brands(self, brand1, list_brand2):
        """
        Used when merging 2 datasets
        Args:
            brand1 : brand to merge : str
            list_brand2: list of brand to check: str
        Return: 
            True, index: if the brand1 already in list_brand2, return the index in list_brand2
            Flase, -1  : if brand1 is not a part of list_brand2
        """
        source_class = self.check_decoded(brand1).lower().replace("_", "").replace("-", "")
        target_classes = [self.check_decoded(t).lower().replace("_", "").replace("-", "") for t in list_brand2]
        if source_class in target_classes:
            return True, list_brand2[target_classes.index(source_class)]
        else:
            return False, -1
    
# CTG_PATH = "data_backup_copy/LogoDet-3K"
# Sherrif = spellingPolice()
# categories = list(glob.iglob(os.path.join("data_backup_copy/LogoDet-3K", "*")))
# count = 0
# for category in categories:
#     Sherrif.check_duplicated(category)
#     classes = os.listdir(category)
#     for c in classes:
#         cls_path = os.path.join(os.path.join(CTG_PATH, category.split(os.path.sep)[-1]), c)
#         renamed = Sherrif.check_decoded(c)
#         renamed = Sherrif.check_special_characters(renamed)
#         if c != renamed:
#             os.rename(cls_path, os.path.join(os.path.join(CTG_PATH, category.split(os.path.sep)[-1]), renamed))
#             count += 1
# print("{} folders have been renamed".format(count))


