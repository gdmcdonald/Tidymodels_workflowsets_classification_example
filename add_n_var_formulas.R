add_n_var_formulas <- function(y_var, x_vars, data, n = 2, include_base = T, include_full = T){

  potential_x_vars <-
    names(data) %>% #variable names
    {.[!. %in% c(y_var, x_vars)]} #remove y and existing x vars

  combos <-
    potential_x_vars %>%
    DescTools::CombSet(n, repl=FALSE, ord=FALSE) %>%
    as_tibble() %>%
    unite(col = combination, sep = " + ") %>%
    pull(combination, name = combination)

  if (include_base) {combos[length(combos)+1]<-"1"}
  if (include_full) {combos[length(combos)+1]<-"."}

  base_pred <- paste(x_vars, collapse = " + ")

  new_formulas <-
    combos %>%
    {paste(y_var, " ~ ",paste(base_pred,., sep = " + "))} %>%  #make formula strings
    purrr::map(as.formula) %>% #make formula
    purrr::map(workflowsets:::rm_formula_env)

  names(new_formulas) <- combos

  if(include_base){ names(new_formulas)[names(new_formulas) == "1"] <- "Base Model" }
  if(include_full){names(new_formulas)[names(new_formulas) == "."] <- "All Predictors"}

  new_formulas
}
