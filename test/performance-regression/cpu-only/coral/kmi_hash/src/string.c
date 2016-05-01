/** KMI string functions.
 */
#include "kmi.h"

static const char KMI_char_mask[] = {
    0xff,
    0xc0,
    0x30,
    0xc,
};

/** Comprare the prefix of two strings in binary form.
 * Note the prefix_len is the original string length.
 * Because both the prefix and the read are stored in binary form,
 * so a simple string compare will not meet the situation when
 * the length of prefix can not be divided exactly by 4.
 * It returns an integer less than, equal to, or greater than zero 
 * if prefix is found, respectively, to be less than, to match, 
 * or be greater than read.
 */
int KMI_prefix_cmp(char *prefix, char *read, int prefix_len)
{
    int remainder = prefix_len % 4;
    int len = prefix_len / 4;
    int rc = memcmp(prefix, read, len);
    if (rc != 0)
        return rc;
    if (remainder == 0)
        return 0;
    char p = KMI_char_mask[remainder] & prefix[len];
    char r = KMI_char_mask[remainder] & read[len];
    return p - r;
}

/** Convert nucleotide sequence from letter to integer representation.
 */
char KMI_str_int2nt(int x)
{
    switch (x) {
    case 0:
        return 'A';
    case 1:
        return 'C';
    case 2:
        return 'G';
    case 3:
        return 'T';
    default:
        return 'N';
    }
}

/** Convert a char* array to a long long integer.
 */
long long KMI_char2longlong(char *str, int str_len)
{
    if (str_len > 32)
        str_len = 32;
    if (str_len < 0) {
        printf("string length should be larger than 0\n");
        return 0;
    }

    long long res = 0;
    int i;
    int remainder = str_len % 4;
    int len = str_len / 4;
    for (i = 0; i < len - 1; i++) {
        res |= (long long) str[i] & 0xff;
        res = res << 8;
    }
    res |= (long long) str[i] & 0xff;
    if (remainder != 0) {
        res = res << (remainder * 2);
        res |= ((long long) str[len] & 0xff) >> (8 - remainder * 2);
    }

    return res;
}

/** Convert a char* array to a long integer.
 */
long KMI_char2long(char *str, int str_len)
{
    if (str_len > 32)
        str_len = 32;
    if (str_len < 0) {
        printf("string length should be larger than 0\n");
        return 0;
    }

    long res = 0;
    int i;
    int remainder = str_len % 4;
    int len = str_len / 4;
    for (i = 0; i < len - 1; i++) {
        res |= (long) str[i] & 0xff;
        res = res << 8;
    }
    res |= (long) str[i] & 0xff;
    if (remainder != 0) {
        res = res << (remainder * 2);
        res |= ((long) str[len] & 0xff) >> (8 - remainder * 2);
    }

    return res;
}

/** Convert nucleotide sequence from letter to xor integer representation.
 * A XOR T = 01
 * C XOR G = 00
 */
inline int KMI_str_xor_nt2int(char c)
{
    switch (c) {
    case 'A':
    case 'a':
    case 'T':
    case 't':
        return 1;
    case 'C':
    case 'c':
    case 'G':
    case 'g':
        return 0;
    default:
        return -1;
    }
}

/** Convert nucleotide sequence from letter to integer representation.
 */
inline int KMI_str_nt2int(char str)
{
    switch (str) {
    case 'A':
    case 'a':
        return 0;
    case 'C':
    case 'c':
        return 1;
    case 'G':
    case 'g':
        return 2;
    case 'T':
    case 't':
        return 3;
    default:
        return -1;
    }
}

/** Convert a binary form string back to letters.
 * Note str_len here is the original length of string.
 */
int KMI_str_bin2ltr(char *str_bin, long str_len, char **str)
{
    *str = (char *) kmi_malloc(str_len + 1);
    long i, j;
    for (i = 0; i < str_len; i++) {
        j = i / 4;
        int bitpos = (i % 4) * 2;
        int x = (int) ((str_bin[j] >> (6 - bitpos)) & 0x3);
        char c = KMI_str_int2nt(x);
        (*str)[i] = c;
    }
    (*str)[i] = '\0';
    return 0;
}

int KMI_str_bin2ltr2(char *str_bin, long str_len, char *str)
{
    long i, j;
    for (i = 0; i < str_len; i++) {
        j = i / 4;
        int bitpos = (i % 4) * 2;
        int x = (int) ((str_bin[j] >> (6 - bitpos)) & 0x3);
        char c = KMI_str_int2nt(x);
        str[i] = c;
    }
    return 0;
}


void debug_print_bin_num(char *str_bin, long str_len)
{
    printf("0b");
    long i, j;
    for (i = 0; i < str_len; i++) {
        j = i / 4;
        int bitpos = (i % 4) * 2;
        int x = (int) ((str_bin[j] >> (6 - bitpos)) & 0x3);
        if (x == 0)
            printf("%d%d", 0, 0);
        else if (x == 1)
            printf("%d%d", 0, 1);
        else if (x == 2)
            printf("%d%d", 1, 0);
        else
            printf("%d%d", 1, 1);
    }
}

void debug_print_bin_str(char *str_bin, long str_len)
{
    long i, j;
    for (i = 0; i < str_len; i++) {
        j = i / 4;
        int bitpos = (i % 4) * 2;
        int x = (int) ((str_bin[j] >> (6 - bitpos)) & 0x3);
        char c = KMI_str_int2nt(x);
        printf("%c", c);
    }
}

/* Return a reverse complement letter */
char KMI_str_letter_rc(char c)
{
    switch (c) {
    case 'A':
    case 'a':
        return 'T';
    case 'C':
    case 'c':
        return 'G';
    case 'G':
    case 'g':
        return 'C';
    case 'T':
    case 't':
        return 'A';
    default:
        return 'N';
    }
}

/** Convert a string to its reverse complement string.
 * @input str: input string
 * @input str_len: the length of input string
 * @output str_rc: output reverse complement string
 */
int KMI_str_rc(char *str, long str_len, char **str_rc)
{
    *str_rc = (char *) kmi_malloc(str_len);
    long i;
    for (i = 0; i < str_len; i++) {
        (*str_rc)[i] = KMI_str_letter_rc(str[str_len - i - 1]);
    }
}

/** Convert a string to a more compact binary form.
 * Use 2 bits for each base letter, {0,1,2,3} for {A,C,G,T}.
 * Another additional 8 bits is used for adjacent edges.
 * Note: remember to free the memory after using str_bin.
 */
int KMI_str_ltr2bin(char *str, int str_len, char **str_bin, int *str_bin_len)
{
    *str_bin_len = KMI_BIN_STR_LEN(str_len);
    *str_bin = (char *) kmi_malloc((*str_bin_len) * sizeof(char));
    int i, j;
    for (i = 0; i < str_len; i++) {
        j = i / 4;
        int bitpos = (i % 4) * 2;
        int tmp = KMI_str_nt2int(str[i]);
        if (bitpos == 0)
            (*str_bin)[j] = (char) (tmp << (6 - bitpos));
        else
            (*str_bin)[j] |= (char) (tmp << (6 - bitpos));
    }
    return 0;
}

/** Convert a string to a more compact binary form.
 * Use 2 bits for each base letter, {0,1,2,3} for {A,C,G,T}.
 */
void KMI_str2binchar(const char * str, int str_len, char * str_bin)
{
	int i;
    for (i = 0; i < str_len; i++) {
        int j = i / 4;
        int bitpos = (i % 4) * 2;
        int tmp = KMI_str_nt2int(str[i]);
        if (bitpos == 0)
            str_bin[j] = (char) (tmp << (6 - bitpos));
        else
            str_bin[j] |= (char) (tmp << (6 - bitpos));
    }
}
