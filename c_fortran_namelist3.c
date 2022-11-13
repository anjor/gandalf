/* Fortran Namelist Reader for C
 * Written by Edmund Highcock
 * edmundhighcock@sourceforge.net
 *
 * This is free software released 
 * under the GPL v3 */


#include <stdio.h> 
#include <stdlib.h>
#include <regex.h>
#include <string.h>

int FNR_DEBUG=0;

void fnr_error_message(char * message, int exit)
{
	printf("%s\n", message);
	if (exit) abort();
}

void fnr_debug_write(char * message)
{
	if (FNR_DEBUG) printf("%s\n", message);
}

struct fnr_struct
{
	int n_namelists;
	char ** namelist_names;
	int * namelist_sizes;
	char *** variable_names;
	char *** variable_values;
	/*void * template_ptr;*/
	/*int check_template;*/
};

int fnr_file_size(FILE * fp)
{
		/*Get file size*/
		int sz;
		fseek(fp, 0L, SEEK_END);
		sz = ftell(fp);

		/*Seek back to the beginning:*/
		fseek(fp, 0L, SEEK_SET);
		return sz;
}
void fnr_read_file(char * fname, char ** text_ptr)
{
	/*FILE * fp=fopen("my_file.txt", "r");*/

  	FILE * fp=fopen(fname, "r");
	  
		int sz = fnr_file_size(fp); /* File size*/

		/*printf("Size was %d\n", sz);*/

		*text_ptr = (char *)malloc((sz)*sizeof(char));

		char *text = *text_ptr;
		

    int i=0;
    while(!feof(fp)) {
			/*printf("I is %d\n", i);*/
			text[i++] = fgetc(fp);
			/*printf("reading");*/
		}
    text[i-1]='\0';
		fclose(fp);
}
int fnr_count_matches(char * text, regex_t regex){

	int reti;
	if (FNR_DEBUG) printf("Marker A4\n");

	int location = 0;
	int text_length = strlen(text);
	int nmatches=0;
	if (FNR_DEBUG) printf("Marker D1\n");

	size_t nmatch = 1;
	regmatch_t  length_match[1];
	while (location < text_length - 1){
		if (FNR_DEBUG) printf("Location %d\n", location);
		reti = regexec(&regex, &text[location], nmatch, length_match, 0);
		if (!reti)
		{
			/*printf("First letter %s", &text[location + length_match[0].rm_so + 4]);*/
			location = location +  length_match[0].rm_eo ;
			nmatches += 1;
		}
		if (reti) break;
	}
	if (FNR_DEBUG) printf("Matches was %d\n", nmatches);
	return nmatches;
}


int fnr_count_namelists(char * text)
{

	if (FNR_DEBUG) printf("void fnr_count_namelists; string is %s\n", text);
	regex_t regex;
	int reti;
	int nmatches;
 

/* Compile regular expression */
	/*reti = regcomp(&regex, "&[_[:alnum:]]\\+\\?\n", 0);*/
	/*reti = regcomp(&regex, "&[_[:alnum:]]+[[:blank:]]", REG_EXTENDED);*/
	reti = regcomp(&regex, "^&[_[:alnum:]]+[\n\r[:blank:]]", REG_EXTENDED|REG_NEWLINE);
	if( reti ){ fprintf(stderr, "Could not compile regex\n"); exit(1); }
	nmatches = fnr_count_matches(text, regex);
  regfree(&regex);
	if (FNR_DEBUG) printf("Marker A6\n");
	return nmatches;

}

int fnr_count_variables(char * text)
{

	if (FNR_DEBUG) printf("void fnr_count_variables; string is %s\n", text);
	regex_t regex;
	int reti;
	int nmatches;
 

/* Compile regular expression */
	reti = regcomp(&regex, "^[[:space:]]*[_[:alnum:]]+([[:blank:]]|=)", REG_EXTENDED|REG_NEWLINE);
	if( reti ){ fprintf(stderr, "Could not compile regex\n"); exit(1); }
	nmatches = fnr_count_matches(text, regex);
  regfree(&regex);
	if (FNR_DEBUG) printf("Marker A8\n");
	return nmatches;

}

void fnr_match_namelists(char * text, char ** namelist_names, char ** namelist_texts)
{

	if (FNR_DEBUG) printf("void fnr_match_namelists\n");
	regex_t regex;
	int reti;
	int location = 0;
	int text_length = strlen(text);
	int nmatches=0;
	size_t nmatch = 3;
	regmatch_t  length_match[3];
	int name_size, ntext_size;
 

/* Compile regular expression */
//	reti = regcomp(&regex, "^&([_[:alnum:]]+)([[:blank:]\n\r](!.*/.*(\r|\n)|[^/]|(\r|\n))+)^/", REG_EXTENDED|REG_NEWLINE);
	//reti = regcomp(&regex, "[\n\r][[:blank:]]*&([_[:alnum:]]+)([[:blank:]\n\r](!.*[\r\n]|[^/]|[\r\n])+)^/", REG_EXTENDED|REG_NEWLINE);
	//reti = regcomp(&regex, "^[[:blank:]]*&([_[:alnum:]]+)([[:blank:]\n\r](!.*|[^/\r\n]|[\r\n]+([^/\n\r]|[\n\r][^/]))+)[\r\n]/", REG_EXTENDED|REG_NEWLINE);
	/*reti = regcomp(&regex, "^[[:blank:]]*&([_[:alnum:]]+)([[:blank:]\n\r]([^\n\r]/|(\n|\r\n)[^/\n\r]|[^\n\r/])+)(\n|\r\n)/", REG_EXTENDED|REG_NEWLINE);*/
	reti = regcomp(&regex, "^[[:blank:]]*&([_[:alnum:]]+)([[:blank:]\n\r]([^\n\r]/|(\n|\r\n)|[^\n\r/])+)^/", REG_EXTENDED|REG_NEWLINE);
	/*reti = regcomp(&regex, "^[[:blank:]]*&([_[:alnum:]]+)([[:blank:]\n\r]([^\n\r]/|[^/])+)(\n|\r\n)/", REG_EXTENDED|REG_NEWLINE);*/
//	reti = regcomp(&regex, "[\n\r][[:blank:]]*&([_[:alnum:]]+)([[:blank:]\n\r](.|[\r\n])+)^/", REG_EXTENDED|REG_NEWLINE);
	/*reti = regcomp(&regex, "&[_[:alnum:]]\\+\\?\n", 0);*/
	if( reti ){ fprintf(stderr, "Could not compile regex for matching namelist names and texts\n"); exit(1); }
  int i = 0;

	while (location < text_length - 1){
		if (FNR_DEBUG) printf("Location %d\n", location);
		reti = regexec(&regex, &text[location], nmatch, length_match, 0);
		char ** ntexts = namelist_texts;
		char ** nnames = namelist_names;
		if (!reti)
		{

			/*Assign namelist name*/
			name_size = length_match[1].rm_eo - length_match[1].rm_so + 1;
			nnames[i] = (char *)malloc(name_size*sizeof(char));
			strncpy(nnames[i], &text[location+length_match[1].rm_so], name_size-1);
			nnames[i][name_size - 1 ] = '\0';

			/*Assign namelist text*/
			ntext_size = length_match[2].rm_eo - length_match[2].rm_so + 1;
			ntexts[i] = (char *)malloc(ntext_size*sizeof(char));
			strncpy(ntexts[i], &text[location+length_match[2].rm_so], ntext_size-1);
			ntexts[i][ntext_size - 1] = '\0';

			/*variable_text_size = length_match[2].rm_eo - length_match[2].rm_so;*/

			if (FNR_DEBUG) printf("begin %d, end %d, Size %d, Name %s\n", length_match[1].rm_so, length_match[1].rm_eo, name_size, nnames[i]);
			if (FNR_DEBUG) printf("begin %d, end %d, Size %d, Name %s\n", length_match[2].rm_so, length_match[2].rm_eo, ntext_size, ntexts[i]);
			/*nmatches++;*/
			i++;

			location = location +  length_match[0].rm_eo;
		}
		if (reti) {
			if (FNR_DEBUG) printf("Finished matching namelists\n");
			break;
		}
	}
	/*printf("F*/

	regfree(&regex);
}

void fnr_match_variables(char * text, char ** variable_names, char ** variable_values)
{

	if (FNR_DEBUG) printf("void fnr_match_variables\n");
	regex_t regex;
	int reti;
	int location = 0;
	int text_length = strlen(text);
	int nmatches=0;
	size_t nmatch = 6;
	regmatch_t  length_match[6];
	int name_size, value_size;
 

/* Compile regular expression */
	/*reti = regcomp(&regex, "(^|\n)[[:space:]]*([_[:alnum:]]+)([[:blank:]]|=)[[:space:]]*=[[:space:]]*(\"([^\"]|\\\\|\\\")+\"|'([[^']|\\\\|\\')+'|[[:alnum:].+-]+)([[:blank:]\r\n]|!)", REG_EXTENDED|REG_NEWLINE);*/
	reti = regcomp(&regex, "(^|\n|\r)[[:space:]]*([_[:alnum:]]+)([[:blank:]]+=|=)[[:blank:]]*(\"([^\"]|\\\\|\\\")+\"|'([^']|\\\\|\\')+'|[[:alnum:].+-]+)([[:blank:]\r\n]|!)", REG_EXTENDED|REG_NEWLINE);
	if( reti ){ fprintf(stderr, "Could not compile regex for matching namelist names and texts\n"); exit(1); }
  int i = 0;

	if (FNR_DEBUG) printf ("Finished making regex; MARKER D1\n");
	if (FNR_DEBUG) printf ("location, %d, text_length, %d, text %s\n", location, text_length, text);
	while (location < text_length - 1){
		if (FNR_DEBUG) printf("Location %d\n", location);
		reti = regexec(&regex, &text[location], nmatch, length_match, 0);
		char ** vnames = variable_names;
		char ** vvalues = variable_values;
		if (!reti)
		{

			/*Assign variable name*/
			int a = 2;
			int b = 4;
			name_size = length_match[a].rm_eo - length_match[a].rm_so + 1;
			vnames[i] = (char *)malloc(name_size*sizeof(char));
			strncpy(vnames[i], &text[location+length_match[a].rm_so], name_size-1);
			vnames[i][name_size - 1 ] = '\0';

			/*Assign variable value*/
			value_size = length_match[b].rm_eo - length_match[b].rm_so + 1;
			vvalues[i] = (char *)malloc(value_size*sizeof(char));
			strncpy(vvalues[i], &text[location+length_match[b].rm_so], value_size-1);
			vvalues[i][value_size - 1] = '\0';

			/*variable_text_size = length_match[2].rm_eo - length_match[2].rm_so;*/

			if (FNR_DEBUG) printf("begin %d, end %d, Size %d, Name %s\n", length_match[2].rm_so, length_match[2].rm_eo, name_size, vnames[i]);
			if (FNR_DEBUG) printf("begin %d, end %d, Size %d, Name %s\n", length_match[3].rm_so, length_match[3].rm_eo, value_size, vvalues[i]);
			/*nmatches++;*/
			i++;

			location = location +  length_match[0].rm_eo;
		}
		if (reti) {
			if (FNR_DEBUG) printf("Finished matching variables\n");
			break;
		}
	}
	regfree(&regex);
}

struct fnr_struct fnr_read_namelist_string(char * file_string)
{
	struct fnr_struct namelist_struct;
	char  ** namelist_texts;
	if (FNR_DEBUG) printf("The string to be read is %s\n\n", file_string);
	/*fnr_match_namelists(file_string, namelist_struct.namelist_names, &namelist_texts);*/

	/* Count the namelists and allocate the namelists arrays accordingly */
	namelist_struct.n_namelists = fnr_count_namelists(file_string);

	namelist_struct.namelist_names  =  (char **)malloc(namelist_struct.n_namelists*sizeof(char *));
	namelist_struct.namelist_sizes  =   (int *)malloc(namelist_struct.n_namelists*sizeof(int));
	namelist_struct.variable_names  = (char ***)malloc(namelist_struct.n_namelists*sizeof(char **));
	namelist_struct.variable_values = (char ***)malloc(namelist_struct.n_namelists*sizeof(char **));
	namelist_texts                  =  (char **)malloc(namelist_struct.n_namelists*sizeof(char *));

	/* Match all the namelists, put their names into namelist names and
	 * their content into namelist_texts*/
	fnr_match_namelists(file_string, namelist_struct.namelist_names, namelist_texts);
	int i,j;
	int nvars;
	for (i=0; i < namelist_struct.n_namelists; i++)
	{
		if (FNR_DEBUG) printf("Analysing namelist %d, called %s\n", i, namelist_struct.namelist_names[i]);
		nvars = namelist_struct.namelist_sizes[i] = fnr_count_variables(namelist_texts[i]);
		namelist_struct.variable_names[i] = (char **)malloc(nvars*sizeof(char *));
		namelist_struct.variable_values[i] = (char **)malloc(nvars*sizeof(char *));
		fnr_match_variables(namelist_texts[i], namelist_struct.variable_names[i], namelist_struct.variable_values[i]); 
		free(namelist_texts[i]);

	}


	free(namelist_texts);

	return namelist_struct;
};
struct fnr_struct fnr_read_namelist_file(char * file_name)
{
	char * file_string;
	/*printf("Marker A1\n");*/
	fnr_read_file(file_name, &file_string);
	if (FNR_DEBUG) printf("The string read was: \n%s\n", file_string);
	struct fnr_struct namelist_struct = fnr_read_namelist_string(file_string);
	return namelist_struct;
}

void fnr_free(struct fnr_struct * namelist_struct){
	int i,j;
	for (i=0; i < namelist_struct->n_namelists; i++)
	{
		for (j=0; j < namelist_struct->namelist_sizes[i];j++){
			free(namelist_struct->variable_names[i][j]);
			free(namelist_struct->variable_values[i][j]);
		}
		free(namelist_struct->variable_names[i]);
		free(namelist_struct->variable_values[i]);
		free(namelist_struct->namelist_names[i]);
	}
	free(namelist_struct->namelist_sizes);
	free(namelist_struct->namelist_names);
	free(namelist_struct->variable_names);
	free(namelist_struct->variable_values);
}


int FNR_NAMELIST_NOT_FOUND=1;
int FNR_VARIABLE_NOT_FOUND=2;
int FNR_VARIABLE_SSCANF_ERROR=3;
int FNR_NAMELIST_NOT_IN_TEMPLATE=4;
int FNR_VARIABLE_NOT_IN_TEMPLATE=5;

int fnr_abort_on_error;
int fnr_abort_if_missing;

/* Defaults */
/*fnr_abort_on_error=1;*/
/*fnr_abort_if_missing=0;*/

void fnr_check_rvalue(const char * namelist, const char * variable, int rvalue)
{
	if (FNR_DEBUG) printf("rvalue, %d, fnr_abort_if_missing, %d\n", rvalue, fnr_abort_if_missing);
	if (!rvalue) return;
	if (fnr_abort_on_error && rvalue == FNR_VARIABLE_SSCANF_ERROR) 
	{
		printf("Error in namelist %s, variable %s\n", namelist, variable);
		abort();
	}
	if (fnr_abort_if_missing && rvalue == FNR_NAMELIST_NOT_FOUND) 
	{
		printf("Missing  namelist %s\n",  namelist);
		abort();
	}
	if (fnr_abort_if_missing && rvalue == FNR_VARIABLE_NOT_FOUND) 
	{
		printf("Missing variable %s in namelist %s\n", variable, namelist);
		abort();
	}
	if (rvalue == FNR_NAMELIST_NOT_IN_TEMPLATE)
	{
		printf("Namelist %s is not in template (i.e. it is not a valid namelist).\n", namelist);
		abort();
	}
	if (rvalue == FNR_VARIABLE_NOT_IN_TEMPLATE)
	{
		printf("Variable %s in namelist %s is not in template (i.e. it is not a valid variable).\n", variable, namelist);
		abort();
	}


}

int fnr_get_string_no_test(struct fnr_struct * namelist_struct, const char * namelist, const char * variable, char ** value)
	/*{*/
	/*int check_template = 0;*/
	/*struct fnr_struct * dummy;*/
	/*return fnr_get_string(namelist_struct, namelist, variable, value, check_template, dummy);*/
	/*}*/

	/*int fnr_get_string(struct fnr_struct * namelist_struct, const char * namelist, const char * variable, char ** value, const int check_template, const struct fnr_struct * namelist_template)*/
{
	int i,j;
	int found_namelist = 0;
	int found_variable = 0;
	int rvalue = 0;
	for (i=0;i<namelist_struct->n_namelists; i++)
	{
 	 if (!strcmp(namelist_struct->namelist_names[i], namelist) )
	 {
		 found_namelist = 1;	
		 break;
	 }
	}
	if (!found_namelist) rvalue =  FNR_NAMELIST_NOT_FOUND;
	if (found_namelist)
	{
	if (FNR_DEBUG) printf("Found namelist %s, size: %d\n", namelist_struct->namelist_names[i], namelist_struct->namelist_sizes[i]);
		for (j=namelist_struct->namelist_sizes[i]-1;j>-1;j--) 
			/* Must take the last specification of the variable*/
		{
			if (FNR_DEBUG) printf("Marker C2; %d  %s\n", j, namelist_struct->variable_names[i][j]);
		 if (!strcmp(namelist_struct->variable_names[i][j], variable))
		 {
			 found_variable = 1;	
			 break;
		 }
		}
		if (!found_variable) rvalue = FNR_VARIABLE_NOT_FOUND;
		if (FNR_DEBUG) printf("Marker C3\n");
		if (found_variable)
		{
			if (FNR_DEBUG) printf("Found variable %s\n", variable);
			char * v = namelist_struct->variable_values[i][j];
			if (FNR_DEBUG) printf("Found value %s\n", v);
			char * dq = "\"";
			char * sq = "'";
			if (v[0] == dq[0] || v[0] == sq[0])
			{
				if (FNR_DEBUG) printf("Value was a string \n");
				*value = (char *)malloc((strlen(v)-1)*sizeof(char));
				char * val = *value;
				if (FNR_DEBUG) printf("Allocated \n");
				strncpy(val, &v[1], strlen(v)-2);
				if (FNR_DEBUG) printf("Copied: %s \n", val);
				val[strlen(v)-2] = '\0';
				if (FNR_DEBUG) printf("Terminated: %s \n", val);
			}
			else 
			{
				if (FNR_DEBUG) printf("MARKER D4; Length of string %d\n", strlen(v));
				*value = (char *)malloc((strlen(v)+1)*sizeof(char));
				if (FNR_DEBUG) printf("MARKER D5\n");
				strcpy(*value, v);
			}
		}
	}
	return rvalue;
}

int fnr_get_string(struct fnr_struct * namelist_struct, const char * namelist, const char * variable, char ** value)
{
  if (FNR_DEBUG) printf("Getting string no test, namelist %s, variable %s\n", namelist, variable);
	int rvalue = fnr_get_string_no_test(namelist_struct, namelist, variable, value);
	fnr_check_rvalue(namelist, variable, rvalue);
	return rvalue;
}

int fnr_get_int(struct fnr_struct * namelist_struct, const char * namelist, const char * variable, int * value)
{
	char * str_value;
	int scfrvalue=0;
	int rvalue;
	rvalue = fnr_get_string(namelist_struct, namelist, variable, &str_value);
	/*if (rvalue) return rvalue;*/
	if (FNR_DEBUG) printf("Str value was %s\n", str_value);
	if (!rvalue) scfrvalue = sscanf(str_value, "%d", value);
	if (FNR_DEBUG) printf("rvalue was %d\n, int is %d\n", rvalue, *value);
	if (!rvalue && !scfrvalue) rvalue = FNR_VARIABLE_SSCANF_ERROR;
	/*else rvalue = 0;*/
	fnr_check_rvalue(namelist, variable, rvalue);
	return rvalue;
}

int fnr_get_bool(struct fnr_struct * namelist_struct, const char * namelist, const char * variable, int * value)
{
	char * str_value;
	int scfrvalue=0;
	int rvalue;
	rvalue = fnr_get_string(namelist_struct, namelist, variable, &str_value);
	/*if (rvalue) return rvalue;*/
	if (FNR_DEBUG) printf("Str value was %s\n", str_value);
	if (!rvalue) {

		regex_t regex_true, regex_false;
		int reti;
	 

	/* Compile regular expression */
		reti = regcomp(&regex_true, "^(t|\\.true\\.)$", REG_ICASE|REG_EXTENDED);
		if( reti ){ fprintf(stderr, "Could not compile regex_true\n"); exit(1); }
		reti = regcomp(&regex_false, "^(f|\\.false\\.)$", REG_ICASE|REG_EXTENDED);
		if( reti ){ fprintf(stderr, "Could not compile regex_false\n"); exit(1); }
		reti = regexec(&regex_true, str_value, 0, NULL, 0);
		if (!reti) *value = 1;
		else 
		{
			if (FNR_DEBUG) printf("Not True\n");
			reti = regexec(&regex_false, str_value, 0, NULL, 0);
			if (!reti) *value = 0;
			else rvalue=FNR_VARIABLE_SSCANF_ERROR;
		}
		regfree(&regex_true);
		regfree(&regex_false);
		if (FNR_DEBUG) printf("Marker A6\n");
	}
	if (FNR_DEBUG) printf("rvalue was %d\n, int is %d\n", rvalue, *value);
	fnr_check_rvalue(namelist, variable, rvalue);
	return rvalue;
}

int fnr_get_float(struct fnr_struct * namelist_struct, const char * namelist, const char * variable, float * value)
{
	char * str_value;
	int rvalue;
	int scfrvalue=0;
	rvalue = fnr_get_string(namelist_struct, namelist, variable, &str_value);
	/*if (rvalue) return rvalue;*/
	if (FNR_DEBUG) printf("Str value was %s\n", str_value);
	if (!rvalue) scfrvalue = sscanf(str_value, "%f", value);
	if (FNR_DEBUG) printf("rvalue was %d\n, float is %d\n", rvalue, *value);
	if (!rvalue && !scfrvalue) rvalue = FNR_VARIABLE_SSCANF_ERROR;
	/*else rvalue = 0;*/
	fnr_check_rvalue(namelist, variable, rvalue);
	return rvalue;
}


int fnr_get_double(struct fnr_struct * namelist_struct, const char * namelist, const char * variable, double * value)
{
	char * str_value;
	int rvalue;
	int scfrvalue=0;
	rvalue = fnr_get_string(namelist_struct, namelist, variable, &str_value);
	/*if (rvalue) return rvalue;*/
	if (FNR_DEBUG) printf("Str value was %s\n, rvalue %d", str_value, rvalue);
	if (!rvalue) scfrvalue = sscanf(str_value, "%lf", value);
	if (FNR_DEBUG) printf("scrvalue was %d\n, double is %f\n", rvalue, *value);
	if (!rvalue && !scfrvalue) rvalue = FNR_VARIABLE_SSCANF_ERROR;
	/*else rvalue = 0;*/
	fnr_check_rvalue(namelist, variable, rvalue);
	return rvalue;
}

void fnr_check_namelist_against_template(struct fnr_struct * namelist_struct, struct fnr_struct * template_struct)
{
	int i,j;
	for (i=0; i < namelist_struct->n_namelists; i++)
	{
		for (j=0; j < namelist_struct->namelist_sizes[i];j++)
		{
			char * dummy;
		  int template_rvalue	= fnr_get_string_no_test(template_struct, namelist_struct->namelist_names[i], namelist_struct->variable_names[i][j], &dummy);
			if (FNR_DEBUG) printf("Template return value was %d\n", template_rvalue);
			/*int old_abort = fnr_abort_if_missing;*/
			/*fnr_abort_if_missing = 0;*/
			int rvalue = template_rvalue;
			if (template_rvalue == FNR_NAMELIST_NOT_FOUND) rvalue = FNR_NAMELIST_NOT_IN_TEMPLATE;
			if (template_rvalue == FNR_VARIABLE_NOT_FOUND) rvalue = FNR_VARIABLE_NOT_IN_TEMPLATE;
			fnr_check_rvalue(namelist_struct->namelist_names[i], namelist_struct->variable_names[i][j], rvalue);
		}
	}
}


char * FNR_TEMPLATE_STRING =  "\n\
\n\
&my_namelist\n\
	beta = \"This is some help for beta\"\n\
/\n\
\n\
&my_namelist1\n\
  !asd\n\
	beta = 2.7 ! adadfsa\n\
 asdf = \"xxxsdfa\"\n\
\n\
/\n\
";


// int main (int argc, char ** argv) {
// 	if (argc < 2) fnr_error_message("Please pass the first test input file as the first parameter.", 1);
// 
// 	int noah = 1;
// 
// 	if (noah)
// 	{
// 		struct fnr_struct namelist_struct = fnr_read_namelist_file(argv[1]);
// 		int Nx;
// 		fnr_abort_on_error = 1;
// 		fnr_abort_if_missing = 1;
// 
// 		if(fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "nx", &Nx)) *&Nx=128;
// 		printf("Nx was %d\n", Nx);
// 		return(0);
// 	}
// 
// 	if (argc < 3) fnr_error_message("Please pass the second test input file as the second parameter.", 1);
// 	if (FNR_DEBUG) printf("Read namelist template???\n");
// 
// 	if (!strcmp(argv[1], "help_variable"))
// 	{
// 		if (argc < 4) fnr_error_message("Please pass the namelist as the second parameter and the variable as the third.", 1);
// 		struct fnr_struct template_struct_help = fnr_read_namelist_string(FNR_TEMPLATE_STRING);
// 		if (FNR_DEBUG) printf("Read namelist template successful\n");
// 		fnr_abort_on_error = 1;
// 		fnr_abort_if_missing = 1;
// 		char * help;
// 		if (fnr_get_string(&template_struct_help, argv[2], argv[3], &help))
// 			printf("No help available");
// 		else
// 			printf("%s\n", help);
// 		exit(0);
// 	}
// 
// 	struct fnr_struct namelist_struct = fnr_read_namelist_file(argv[1]);
// 	/*namelist_struct.check_template = 0;*/
// 
// 	fnr_abort_on_error = 1;
// 	fnr_abort_if_missing = 1;
// 
// 	/* String */
// 	/* fnr_get returns 0 if successful */
// 	char * collision_model;
// 	if (fnr_get_string(&namelist_struct, "collisions_knobs", "collision_model", &collision_model))
// 		collision_model = "default";
// 	printf("Collison model was %s\n", collision_model);
// 
// 	/* Integer */
// 	int nx;
// 	if (fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "nx", &nx)) nx = 0;
// 	printf("nx was %d\n", nx);
// 
// 	/* Float */
// 	float g_exb;
// 	if (fnr_get_float(&namelist_struct, "dist_fn_knobs", "g_exb", &g_exb)) g_exb = 4.0;
// 	printf("g_exb was %f\n", g_exb);
// 
// 	/* Double */
// 	double phiinit;
// 	if (fnr_get_double(&namelist_struct, "init_g_knobs", "phiinit", &phiinit)) phiinit = 0.0;
// 	printf("phiinit was %f\n", phiinit);
// 
// 	/*Bool*/
// 	int write_phi_over_time;
// 	if (fnr_get_bool(&namelist_struct, "gs2_diagnostics_knobs", "write_phi_over_time", &write_phi_over_time)) write_phi_over_time = 0;
// 	printf("write_phi_over_time was %d\n", write_phi_over_time);
// 
// 
// 	/*Fails*/
// 	/*if (fnr_get_double(&namelist_struct, "init_g_knobs", "hiinit", &phiinit)) phiinit = 0.0;*/
// 	/*printf("phiinit was %f\n", phiinit);*/
// 
// 	printf("Success!\n");
// 
// 
// 	struct fnr_struct namelist_struct_with_template = fnr_read_namelist_file(argv[2]);
// 	if (FNR_DEBUG) printf("Finished reading second namelist\n\n");
// 	struct fnr_struct template_struct = fnr_read_namelist_string(FNR_TEMPLATE_STRING);
// 
// 	fnr_abort_on_error = 1;
// 	fnr_abort_if_missing = 0;
// 
// 	double beta = 1.2;
// 	if (fnr_get_double(&namelist_struct_with_template, "my_namelist1", "beta", &beta)) beta = 0.5; 
// 	printf("beta was %e\n", beta);
// 
// 	fnr_check_namelist_against_template(&namelist_struct_with_template, &template_struct);
// 
// 	fnr_free(&namelist_struct);
// 	fnr_free(&namelist_struct_with_template);
// 	fnr_free(&template_struct);
// }
// 
// 
// 
// 
// 
