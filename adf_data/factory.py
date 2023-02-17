from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_data.config import census, credit, bank

data = {"census": census_data, "credit": credit_data, "bank": bank_data}
data_config = {"census": census, "credit": credit, "bank": bank}

class DataFactory():
    @staticmethod
    def factory(dataset):
        X, Y, input_shape, nb_classes = data[dataset]()
        config = data_config[dataset]
        return X, Y, input_shape, nb_classes, config
