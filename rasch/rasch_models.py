from .utils import dichotomous, ICC, IIF, TIF


class RaschModel:
    def __init__(self, items):
        self.items = items
        self.itemparameters = None
        self.personparameters = None

    def estimate_parameters(self):
        raise NotImplementedError("This method will be implemented by the subclass")

    def ICCplot(self):
        raise NotImplementedError("This method will be implemented by the subclass")
    
    def IIFplot(self):
        raise NotImplementedError("This method will be implemented by the subclass")
    
    def TIFplot(self):
        raise NotImplementedError("This method will be implemented by the subclass")


class DichotomousRaschModel(RaschModel):
    def estimate_parameters(self):
        difficulty_df, ability_df = dichotomous(self.items)
        self.itemparameters = difficulty_df
        self.personparameters = ability_df
        return (difficulty_df, ability_df)

    def ICCplot(self):
       ICC(self.itemparameters)
    
    def IIFplot(self):
        IIF(self.itemparameters)
    
    def TIFplot(self):
        TIF(self.itemparameters)