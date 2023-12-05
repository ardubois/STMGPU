#define WriteSetSize	2
#define ReadSetSize		2

#define ACTIVE      1
#define COMMITTED   2
#define ABORTED     3

typedef struct Locator_
{
	unsigned int owner;
	int* new_version;
	int* old_version;
} Locator;

typedef struct readSet_
{
	unsigned short size;
    Locator* locator[ReadSetSize];
    unsigned int object[ReadSetSize];
    int* value[ReadSetSize];
} ReadSet;

typedef struct writeSet_
{
    unsigned short size;
    unsigned int object[ReadSetSize];
} WriteSet;

typedef struct TX_Data_
{
    unsigned int tr_id;
    unsigned short next_locator;
    ReadSet read_set;
    WriteSet write_set;
} TX_Data;

typedef struct STMData_
{
	Locator* objects;
    Locator** vboxes;
	unsigned short* tr_state;
    Locator* locators;
    unsigned short num_locators;
    TX_Data* tx_data;
} STMData;

