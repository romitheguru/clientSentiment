import os
import MySQLdb as mdb

connection = mdb.connect('localhost', 'root', 'password')
c = connection.cursor()

# Open and read the file as a single buffer
fd = open('gpu_aml_dump.sql', 'r')
sqlFile = fd.read()
fd.close()

# all SQL commands (split on ';')
sqlCommands = sqlFile.split(';')

# Execute every command from the input file
for command in sqlCommands:
    # This will skip and report errors
    # For example, if the tables do not yet exist, this will skip over
    # the DROP TABLE commands
    try:
        c.execute(command)
    except:
        print "Command skipped"


# For each of the 3 tables, query the database and print the contents
for table in ['account', 'bank', 'billers', 'classification', 'country', 'countrydetails']:


    # Plug in the name of the table into SELECT * query
    result = c.execute("SELECT * FROM %s;" % table);

    # Get all rows.
    rows = result.fetchall();

    # \n represents an end-of-line
    print "\n--- TABLE ", table, "\n"

    # This will print the name of the columns, padding each name up
    # to 22 characters. Note that comma at the end prevents new lines
    for desc in result.description:
        print desc[0].rjust(22, ' '),

    # End the line with column names
    print ""
    for row in rows:
        for value in row:
            # Print each value, padding it up with ' ' to 22 characters on the right
            print str(value).rjust(22, ' '),
        # End the values from the row
        print ""

c.close()
conn.close()