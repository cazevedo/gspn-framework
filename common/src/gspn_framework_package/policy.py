

class Policy(object):
    def __init__(self,  policy_dictionary, places_tuple):
        '''
        :param policy_dictionary: dictionary where key= tuple with the marking; value= dictionary where key is
        transitions and value is the probability of firing
        :param places_tuple: tuple with the order of the places that are represented in policy_dictionary
        '''
        self.__policy_dictionary = policy_dictionary
        self.__places_tuple = places_tuple

    def get_policy_dictionary(self):
        return self.__policy_dictionary

    def get_places_tuple(self):
        return self.__places_tuple
