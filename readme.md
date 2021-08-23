# Fault Injector

The purpose of this project is to implement a fault injector capable of simulating both stuck-at fault and bit-flip faults that affect the weights of a tensorflow network.

# Prerequisites

The following libraries are required to run the script:
* numpy
* pandas
* tensorflow

# Implementation detail 

The folder **FaultInjector** contains the files required ot perform the fault injection campaign. 

In the file **FaultInjectorEngine.py** are implemented the function that are used to perform the bit-flip and the stuck-at fault on the weight.

The file **NetworkFaultInjector.py** contains the base class NetworkFaultInjector that is used to generate, save, and load a fault list, and perform an injection campaign. This class is extended by the **BitFlipFaultInjector** and the **StuckAtFaultInjector** class, that contain the implementation of these functions for the bit-flip and the stuck-at fault.

The folder **RunManager** contains the file **NetworkManager.py** that is used to execute an inference run over a given dataset and save the results. Additionally, this folder contains also the **ImageLoader.py** file that provides some useful function for the dataset manipulation.

# Demo

The file **main.py** contains a simple demo of the tool. There are two main steps: first we execute a **golden run**, that is, we obtain the vector scores from the network in absence of fault. After that, we perform multiple injection campaigns, and for each of them we store the value of the vector scores. These runs are called **faulty runs**. 

Firstly, we create an instance of VGG-16 using the function provided by tensorflow: 

    vgg = VGG16(classifier_activation=None)
    vgg.compile(metrics=['accuracy'])

After that, we initialize the network manager over VGG-16. When we create this instance, we are also attaching a dataset to the network:

    network_manager = NetworkManager(network=vgg, dataset_dir=input_dir)

At this point we can execute the **golden run**, using the function provided by the network manager. In this case we are specifying that we want to save the vector scores into an output_format file and that we are interested only in the top-n elements of the scores. 

    network_manager.run_and_export(run_name=f'vgg_imagenet_inference_result',
                                   output_dir='GoldenRunResults',
                                   top_n=top_n,
                                   output_format=output_format,
                                   pre_processing_function=preprocess_input)

In the following loop we then proceed with the **faulty runs**. The first step is to load the original weights of the network, to obtain a network unaffected by faults. This is done to perform a clean fault injection campaign in the following iterations of the loop.

    network_manager.reset_network()

We then initialize a stuck-at fault fault injector over vgg, specifying a seed:

    fault_injector = StuckAtFaultInjector(vgg, seed)

Given the fault injector, we can perform the fault injection campaign.

    fault_injector.fault_injection_campaign(number_of_faults=number_of_faults,
                                            folder_path='FaultList',
                                            fault_list_length=10000)

This function first looks for a fault list generated with the same seed. If it doesn't exist, it generates it and saves it for later uses. After that, it injects number_of_faults faults into the network weights.

Finally, we can perform an inference over the whole dataset and store result. This is the same function called for the golden run.

    network_manager.run_and_export(run_name=f'vgg_imagenet_{seed}_{number_of_faults}_inference_result',
                                   output_dir='FaultyRunResults',
                                   top_n=top_n,
                                   output_format=output_format,
                                   pre_processing_function=preprocess_input)
