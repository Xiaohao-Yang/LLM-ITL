# [n_epochs, learning rate, batch size]
hyperparamters = {'NVDM_20News':[10, 0.001, 200], # 250
                  'NVDM_R8':[250, 0.0005, 200],
                  'NVDM_DBpedia':[250, 0.001, 200],
                  'NVDM_AGNews':[250, 0.0005, 200],

                  'PLDA_20News': [250, 0.002, 200],
                  'PLDA_R8': [250, 0.0005, 200],
                  'PLDA_DBpedia': [250, 0.002, 200],
                  'PLDA_AGNews': [250, 0.0005, 200],

                  'ETM_20News': [250, 0.005, 2000],
                  'ETM_R8': [250, 0.002, 2000],
                  'ETM_DBpedia': [250, 0.005, 2000],
                  'ETM_AGNews': [250, 0.001, 2000],

                  'ECRTM_20News': [450, 0.002, 200, 100],
                  'ECRTM_R8': [250, 0.002, 200, 300],
                  'ECRTM_DBpedia': [250, 0.0005, 500, 5],
                  'ECRTM_AGNews': [450, 0.0005, 500, 100],

                  'NSTM_20News': [200, 0.0001, 200],
                  'NSTM_R8': [200, 0.0001, 200],
                  'NSTM_DBpedia': [200, 0.0001, 500],
                  'NSTM_AGNews': [200, 0.0001, 1000],

                  'scholar_20News': [550, 0.001, 200],
                  'scholar_R8': [550, 0.001, 1000],
                  'scholar_DBpedia': [550, 0.002, 1000],
                  'scholar_AGNews': [550, 0.001, 500],

                  'clntm_20News': [550, 0.001, 200],
                  'clntm_R8': [550, 0.001, 1000],
                  'clntm_DBpedia': [550, 0.002, 1000],
                  'clntm_AGNews': [550, 0.001, 500],

                  'WeTe_20News': [500, 0.004, 200],
                  'WeTe_R8': [150, 0.002, 200],
                  'WeTe_DBpedia': [250, 0.002, 500],
                  'WeTe_AGNews': [250, 0.0005, 500]
                  }