InfoMine 
========

Identifying Information.dk commenteer gender
############################################


Installation
------------

.. code::

	pip install -r requirements.txt
	python setup.py install

Our data source is a MySQL SQL-file extracted for us, by the very nice information.dk DBA.
We've included this file in the data-folder!

In order to extract the data, you need to import this file into an existing and already configured MySQL DBMS database called "Information".
Login credentials are configured to 'root' with an empty password - this can be changed by altering the mysql_data_extractor.py-file, which were made for this purpose.
Launch the app like this to start extracting:

.. code::
    infominer --mysql ""

For simplicity, we've already done this part and included the generated comments.p-file in the data/ dir.


Usage
-----

Generating the trainingset and create analysis report:

.. code:: python

    >>> python infomine [-v] <comment>


Running the tests:

.. code:: python
    
    >>> python setup.py test
