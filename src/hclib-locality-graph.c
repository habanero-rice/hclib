#include "jsmn.h"
#include "hclib-locality-graph.h"
#include "hclib-rt.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

typedef enum {
    DIVIDE = 0,
    REMAINDER = 1
} LOCALE_OP;

static int string_token_equals(jsmntok_t *token, char *json, const char *str) {
    if (token->type != JSMN_STRING) {
        fprintf(stderr, "token type is not JSMN_STRING when comparing to \"%s\"\n", str);
        exit(1);
    }
    const int token_length = token->end - token->start;
    return strncmp(json + token->start, str, token_length);
}

static hclib_locale *find_matching_locale(jsmntok_t *token, char *json,
        hclib_locale *locales, int nlocales) {
    int i;
    for (i = 0; i < nlocales; i++) {
        if (string_token_equals(token, json, locales[i].lbl) == 0) {
            return locales + i;
        }
    }
    return NULL;
}

static hclib_locale *find_matching_locale_by_str(char *locale_name,
        hclib_locale *locales, int nlocales) {
    int i;
    for (i = 0; i < nlocales; i++) {
        if (strcmp(locale_name, locales[i].lbl) == 0) {
            return locales + i;
        }
    }
    return NULL;
}

static char *get_copy_of_string_token(jsmntok_t *token, char *json) {
    const int token_length = token->end - token->start;
    char *token_str = (char *)malloc(token_length + 1);
    assert(token_str);
    memcpy(token_str, json + token->start, token_length);
    token_str[token_length] = '\0';
    return token_str;
}

static int parse_int_from_primitive(jsmntok_t *token, char *json) {
    assert(token->type == JSMN_PRIMITIVE);
    char *int_str = get_copy_of_string_token(token, json);
    const int val = atoi(int_str);
    free(int_str);
    return val;
}

/*
 * Pulled from http://www.strudel.org.uk/itoa/
 */
static void strreverse(char* begin, char* end) {
    char aux;
    while (end > begin) {
        aux=*end, *end--=*begin, *begin++=aux;
    }
}
	
static void itoa(int value, char* str, int base) {
    static char num[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    char* wstr=str;
    int sign;

    // Validate base
    if (base<2 || base>35){ *wstr='\0'; return; }

    // Take care of sign
    if ((sign=value) < 0) value = -value;

    // Conversion. Number is reversed.
    do *wstr++ = num[value%base]; while(value/=base);
    if(sign<0) *wstr++='-';
    *wstr='\0';

    // Reverse string
    strreverse(str,wstr-1);
}

static int parse_value(char **pc, int worker_id) {
    while (**pc == ' ') {
        *pc = *pc + 1;
    }
    assert(**pc != '\0');

    if (**pc == 'i' && *(*pc + 1) == 'd') {
        // special id value
        *pc = *pc + 2;
        return worker_id;
    } else {
        // Constant integer value
        char buf[1024];
        char *start = *pc;
        while (**pc >= '0' && **pc <= '9') {
            *pc = *pc + 1;
        }
        const size_t length = *pc - start;
        memcpy(buf, start, length);
        buf[length] = '\0';
        return atoi(buf);
    }
}

static LOCALE_OP parse_op(char **pc) {
    while (**pc == ' ') {
        *pc = *pc + 1;
    }
    assert(**pc != '\0');

    if (**pc == '/') {
        *pc = *pc + 1;
        return DIVIDE;
    } else if (**pc == '%') {
        *pc = *pc + 1;
        return REMAINDER;
    } else {
        fprintf(stderr, "Unsupported op character \"%c\"\n", **pc);
        exit(1);
    }
}

static char *interpret_locale(char *locale_name, int worker_id) {
    // TODO we currently just assume 1024 is enough bytes
    char *result = (char *)malloc(1024);
    char buf[1024];

    char *pc = locale_name;
    char *out = result;
    while (*pc != '\0') {
        if (*pc == '$' && *(pc + 1) == '(') {
            pc += 2;

            int value_so_far = parse_value(&pc, worker_id);
            while (*pc != ')') {
                const LOCALE_OP op = parse_op(&pc);
                const int next_value = parse_value(&pc, worker_id);

                switch (op) {
                    case (DIVIDE):
                        value_so_far /= next_value;
                        break;
                    case (REMAINDER):
                        value_so_far = value_so_far % next_value;
                        break;
                    default:
                        fprintf(stderr, "Unimplemented operator %d\n", op);
                        exit(1);
                }
            }
            pc++; // increment past closing paren

            itoa(value_so_far, buf, 10);
            memcpy(out, buf, strlen(buf));
            out += strlen(buf);
        } else {
            *out = *pc;
            pc++; out++;
        }
    }
    *out = '\0';

    return result;
}

static hclib_locality_path *parse_locality_path_from_array(
        jsmntok_t *starting_token, char *json, int worker_id,
        hclib_locale *locales, int nlocales) {
    assert(starting_token->type == JSMN_ARRAY);
    const int path_length = starting_token->size;
    assert(path_length > 0);

    hclib_locality_path *path = (hclib_locality_path *)malloc(
            sizeof(hclib_locality_path));
    assert(path);
    path->locales = (hclib_locale **)malloc(
            path_length * sizeof(hclib_locale *));
    assert(path->locales);
    path->path_length = path_length;

    int i;
    for (i = 0; i < path_length; i++) {
        jsmntok_t *token = starting_token + 1 + i;
        assert(token->type == JSMN_STRING);

        char *str = get_copy_of_string_token(token, json);
        char *interpreted_str = interpret_locale(str, worker_id);
        free(str);

        hclib_locale *locale = find_matching_locale_by_str(interpreted_str,
                locales, nlocales);
        if (!locale) {
            fprintf(stderr, "failed finding locale to match lbl \"%s\"\n", interpreted_str);
            exit(1);
        }
        path->locales[i] = locale;
        free(interpreted_str);
    }

    return path;
}

static int parse_paths(int starting_token_index, int n_paths, char *json,
        jsmntok_t *tokens, hclib_locale *locales, int nlocales,
        hclib_locality_path **worker_paths, jsmntok_t **default_path_token) {
    int i;
    int path_index = starting_token_index;
    *default_path_token = NULL;

    for (i = 0; i < n_paths; i++) {
        if (string_token_equals(tokens + path_index, json, "default") == 0) {
            path_index++;

            assert(tokens[path_index].type == JSMN_ARRAY);
            assert(*default_path_token == NULL);
            *default_path_token = tokens + path_index;

            const int path_length = tokens[path_index].size;
            path_index += 1 + path_length;
        } else {
            const int worker_id = parse_int_from_primitive(
                    tokens + path_index, json);
            path_index++;

            hclib_locality_path *path = parse_locality_path_from_array(
                    tokens + path_index, json, worker_id, locales, nlocales);
            assert(worker_paths[worker_id] == NULL);
            worker_paths[worker_id] = path;
            path_index += 1 + path->path_length;
        }
    }
    return path_index;
}

/*
 * See locality_graphs/davinci.json for an example locality graph.
 */
void load_locality_info(char *filename, int *nworkers_out,
        hclib_locality_graph **graph_out,
        hclib_worker_paths **worker_paths_out) {
    int i;
    jsmn_parser parser;
    jsmn_init(&parser);
    printf("loading locality graph from %s\n", filename);

    FILE *fp = fopen(filename, "r");
    assert(fp);

    fseek(fp, 0L, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    char *json = (char *)malloc(file_size + 1);
    assert(json);
    const size_t nread = fread(json, 1, file_size, fp);
    assert(nread == file_size);
    json[file_size] = '\0';

    // Use jsmn_parse to count the number of tokens in our input JSON
    const int ntokens = jsmn_parse(&parser, json, file_size, NULL, 0);
    assert(ntokens >= 0);

    jsmntok_t *tokens = (jsmntok_t *)malloc(ntokens * sizeof(jsmntok_t));
    assert(tokens);

    // Get the actual tokens from the input JSON
    jsmn_init(&parser);
    const int parsing_err = jsmn_parse(&parser, json, file_size, tokens, ntokens);
    assert(parsing_err >= 0);
    int token_index = 0;

    // Top level object
    assert(tokens[token_index].type == JSMN_OBJECT);
    token_index++;

    // Declarations field of top-level object
    assert(string_token_equals(tokens + token_index, json, "declarations") == 0);
    token_index++;
    assert(tokens[token_index].type == JSMN_ARRAY);
    const int nlocales = tokens[token_index].size;
    token_index++;

    // Initialize locales array from the array of declared locales
    hclib_locale *locales = (hclib_locale *)malloc(nlocales * sizeof(hclib_locale));
    assert(locales);
    for (i = token_index; i < token_index + nlocales; i++) {
        assert(tokens[i].type == JSMN_STRING);
        locales[i - token_index].id = i - token_index;
        locales[i - token_index].lbl = get_copy_of_string_token(tokens + i, json);

        // Verify that this is a unique label across all locales
        int j;
        for (j = 0; j < i - token_index; j++) {
            assert(strcmp(locales[i - token_index].lbl, locales[j].lbl) != 0);
        }
    }
    token_index += nlocales;

    // Initialize a graph object now that we have a list of all locales in the current system
    hclib_locality_graph *graph = (hclib_locality_graph *)malloc(sizeof(hclib_locality_graph));
    assert(graph);
    graph->locales = locales;
    graph->n_locales = nlocales;
    graph->edges = (unsigned *)malloc(nlocales * nlocales * sizeof(unsigned));
    assert(graph->edges);
    memset(graph->edges, 0x00, nlocales * nlocales * sizeof(unsigned));

    // list of reachability edges
    assert(string_token_equals(tokens + token_index, json, "reachability") == 0);
    token_index++;
    assert(tokens[token_index].type == JSMN_ARRAY);
    const int nedges = tokens[token_index].size;
    token_index++;

    int edge_index = token_index;
    /*
     * 3 tokens per edge: a token for the overall array, followed by one token
     * for each of the array's members
     */
    while (edge_index < token_index + nedges * 3) {
        assert(tokens[edge_index].type == JSMN_ARRAY);
        assert(tokens[edge_index].size == 2);
        edge_index++;

        hclib_locale *locale1 = find_matching_locale(tokens + edge_index, json, locales, nlocales);
        if (!locale1) {
            char *lbl = get_copy_of_string_token(tokens + edge_index, json);
            fprintf(stderr, "Locale %s undeclared but referenced in reachability definition\n", lbl);
            exit(1);
        }
        edge_index++;
        hclib_locale *locale2 = find_matching_locale(tokens + edge_index, json, locales, nlocales);
        if (!locale2) {
            char *lbl = get_copy_of_string_token(tokens + edge_index, json);
            fprintf(stderr, "Locale %s undeclared but referenced in reachability definition\n", lbl);
            exit(1);
        }
        edge_index++;

        graph->edges[locale1->id * nlocales + locale2->id] = 1;
        graph->edges[locale2->id * nlocales + locale1->id] = 1;
    }
    token_index = edge_index;

    // Number of workers to create
    assert(string_token_equals(tokens + token_index, json, "nworkers") == 0);
    token_index++;
    const int nworkers = parse_int_from_primitive(tokens + token_index, json);
    token_index++;

    hclib_worker_paths *worker_paths = (hclib_worker_paths *)malloc(nworkers * sizeof(hclib_worker_paths));
    assert(worker_paths);
    memset(worker_paths, 0x00, nworkers * sizeof(hclib_worker_paths));

    hclib_locality_path **worker_pop_paths = (hclib_locality_path **)malloc(
            nworkers * sizeof(hclib_locality_path *));
    assert(worker_pop_paths);
    memset(worker_pop_paths, 0x00, nworkers * sizeof(hclib_locality_path *));
    hclib_locality_path **worker_steal_paths = (hclib_locality_path **)malloc(
            nworkers * sizeof(hclib_locality_path *));
    assert(worker_steal_paths);
    memset(worker_steal_paths, 0x00, nworkers * sizeof(hclib_locality_path *));

    jsmntok_t *default_pop_path_token = NULL;
    jsmntok_t *default_steal_path_token = NULL;

    // List of pop paths for each worker
    assert(string_token_equals(tokens + token_index, json, "pop_paths") == 0);
    token_index++;
    const int n_pop_paths = tokens[token_index].size;
    token_index++;
    assert(n_pop_paths > 0);

    token_index = parse_paths(token_index, n_pop_paths, json, tokens, locales,
            nlocales, worker_pop_paths, &default_pop_path_token);
    assert(default_pop_path_token);

    // List of steal paths for each worker
    assert(string_token_equals(tokens + token_index, json, "steal_paths") == 0);
    token_index++;
    const int n_steal_paths = tokens[token_index].size;
    token_index++;
    assert(n_steal_paths > 0);

    token_index = parse_paths(token_index, n_steal_paths, json, tokens, locales,
            nlocales, worker_steal_paths, &default_steal_path_token);
    assert(default_steal_path_token);

    for (i = 0; i < nworkers; i++) {
        if (worker_pop_paths[i]) {
            worker_paths[i].pop_path = worker_pop_paths[i];
        } else {
            worker_paths[i].pop_path = parse_locality_path_from_array(
                    default_pop_path_token, json, i, locales, nlocales);
        }

        if (worker_steal_paths[i]) {
            worker_paths[i].steal_path = worker_steal_paths[i];
        } else {
            worker_paths[i].steal_path = parse_locality_path_from_array(
                    default_steal_path_token, json, i, locales, nlocales);
        }
    }

    free(worker_pop_paths);
    free(worker_steal_paths);

    /*
     * Final output is the locality graph that depicts the hardware layout of
     * the node (graph) and the set of paths for each worker to traverse when
     * either popping or stealing (worker_paths).
     */
    *nworkers_out = nworkers;
    *graph_out = graph;
    *worker_paths_out = worker_paths;
}

static char *create_heap_allocated_str(const char *s) {
    char *heap_allocated = (char *)malloc(strlen(s) + 1);
    memcpy(heap_allocated, s, strlen(s));
    heap_allocated[strlen(s)] = '\0';
    return heap_allocated;
}

void generate_locality_info(int *nworkers_out,
        hclib_locality_graph **graph_out,
        hclib_worker_paths **worker_paths_out) {
    int i;
    const char *nworkers_str = getenv("HCLIB_WORKERS");
    int nworkers;
    if (nworkers_str) {
        nworkers = atoi(nworkers_str);
    } else {
        nworkers = sysconf(_SC_NPROCESSORS_ONLN);
        fprintf(stderr, "WARNING: HCLIB_WORKERS not provided, running with "
                "default of %u\n", nworkers);
    }

    hclib_locality_graph *graph = (hclib_locality_graph *)malloc(sizeof(hclib_locality_graph));
    assert(graph);
    graph->n_locales = 1 + nworkers;
    graph->locales = (hclib_locale *)malloc(graph->n_locales * sizeof(hclib_locale));
    assert(graph->locales);
    graph->edges = (unsigned *)malloc(graph->n_locales * graph->n_locales * sizeof(unsigned));
    assert(graph->edges);

    hclib_worker_paths *worker_paths = (hclib_worker_paths *)malloc(nworkers * sizeof(hclib_worker_paths));
    assert(worker_paths);

    graph->locales[0].id = 0;
    graph->locales[0].lbl = create_heap_allocated_str("sysmem");
    for (i = 1; i <= nworkers; i++) {
        char buf[128];
        sprintf(buf, "cache%d", i - 1);
        graph->locales[i].id = i;
        graph->locales[i].lbl = create_heap_allocated_str(buf);
        graph->edges[i * graph->n_locales + 0] = 1;
        graph->edges[0 * graph->n_locales + i] = 1;

        worker_paths[i - 1].pop_path = (hclib_locality_path *)malloc(sizeof(hclib_locality_path));
        worker_paths[i - 1].pop_path->path_length = 2;
        worker_paths[i - 1].pop_path->locales = (hclib_locale **)malloc(2 * sizeof(hclib_locale *));
        worker_paths[i - 1].pop_path->locales[0] = graph->locales + i;
        worker_paths[i - 1].pop_path->locales[1] = graph->locales + 0;

        worker_paths[i - 1].steal_path = (hclib_locality_path *)malloc(sizeof(hclib_locality_path));
        worker_paths[i - 1].steal_path->path_length = 2;
        worker_paths[i - 1].steal_path->locales = (hclib_locale **)malloc(2 * sizeof(hclib_locale *));
        worker_paths[i - 1].steal_path->locales[0] = graph->locales + i;
        worker_paths[i - 1].steal_path->locales[1] = graph->locales + 0;
    }

    *nworkers_out = nworkers;
    *graph_out = graph;
    *worker_paths_out = worker_paths;
}

void print_locality_graph(hclib_locality_graph *graph) {
    int i;
    printf("==========================================================\n");
    printf("==== Locality graph %p\n", graph);
    printf("==== # locales = %d\n", graph->n_locales);
    for (i = 0; i < graph->n_locales; i++) {
        hclib_locale *curr = graph->locales + i;
        printf("======== locale %d - %s - connected to ", curr->id, curr->lbl);
        int count_connected = 0;
        int j;
        for (j = 0; j < graph->n_locales; j++) {
            if (graph->edges[i * graph->n_locales + j]) {
                printf("%s ", graph->locales[j].lbl);
                count_connected++;
            }
        }
        if (count_connected == 0) {
            printf("no locales\n");
        }
        printf("\n");
    }
    printf("==========================================================\n");
    printf("\n");
}

void print_worker_paths(hclib_worker_paths *worker_paths, int nworkers) {
    int i, j;
    printf("==========================================================\n");
    printf("==== Worker paths %p for %d workers\n", worker_paths, nworkers);
    for (i = 0; i < nworkers; i++) {
        hclib_worker_paths *curr = worker_paths + i;
        hclib_locality_path *pop = curr->pop_path;
        hclib_locality_path *steal = curr->steal_path;

        printf("======== worker %d\n", i);
        printf("============ pop - ");
        for (j = 0; j < pop->path_length; j++) {
            printf("%s ", pop->locales[j]->lbl);
        }
        printf("\n");
        printf("============ steal - ");
        for (j = 0; j < steal->path_length; j++) {
            printf("%s ", steal->locales[j]->lbl);
        }
        printf("\n");
    }
    printf("==========================================================\n");
    printf("\n");
}
