from .utils import dichotomous, polytomousRasch, ICC, IIF, TIF, ICC_polytomous


class RaschModel:
    def __init__(self, items):
        self.items = items
        self.itemparameters = None
        self.personparameters = None
        self.itemthresholds = None

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

class RatingScaleModel(RaschModel):
    def estimate_parameters(self):
        ability_df, difficulty_df, thresholds = polytomousRasch(self.items)
        self.itemparameters = difficulty_df
        self.personparameters = ability_df    
        self.thresholds = thresholds   
        return (difficulty_df, ability_df, thresholds)
    
    def ICCplot(self, item_name):
        difficulty = self.itemparameters.loc[item_name, "Estimate"] 
        print(difficulty)
        print(self.thresholds)
        ICC_polytomous(thresholds=self.thresholds.iloc[:,1].values,
                       difficulty=difficulty,
                       response_options=len(self.thresholds.iloc[:,1].values),
                       item_index=item_name)