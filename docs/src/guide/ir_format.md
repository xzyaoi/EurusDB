<!--
 Copyright (c) 2021 Xiaozhe Yao et al.
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Intermediate Representation of Models

The machine learning models used as indexes or other components are trained with Python script. In order to use these models in the database implementations, which are usually done in different languages, we convert the learned models into an intermediate representation. After that, the database implementations are responsible for loading and parsing the intermediate representation properly.

The entry file of an intermediate representation is a ```.json``` file that has the following format.

``` json
{
    "meta": {
        "name": "the_model_name"
    },
    "intervals": [0.001,...,0.99], // the intervals must be between 0 to 1.
    "models": [
        {
            "operator":"linear",
            "slope": 1.5,
            "intercept": 0.5
        }
    ]
}
```