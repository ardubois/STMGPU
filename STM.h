#define WriteSetSize	2
#define ReadSetSize		2

#define ACTIVE      1
#define COMMITTED   2
#define ABORTED     3

typedef struct STMData_
{
	Locator *objects;
    Locator *tr_locators;
	ushort *tr_state;
    ushort *committed_tr_state;
} STMData;

typedef struct Locator_
{
	uint owner;
	int new_version;
	int old_version;
} Locator;

typedef struct readSet_
{
	ushort size;
    Locator locator[ReadSetSize];
    uint object[ReadSetSize];
    int value[ReadSetSize];
} readSet;

typedef struct writeSet_
{
    ushort size;
    uint object[ReadSetSize];
}